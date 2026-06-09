import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

from lightgbm import LGBMClassifier

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4


# ============================================================
# 1. CONFIGURATION
# ============================================================
EXCEL_PATH = "MISTA_Analysis.xlsx"   # change if needed
OUTPUT_DIR = Path("avm_ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Optional: harmonize centers if you know some labels represent the same site.
CENTER_MAP = {
    # Keep centers separate unless you explicitly decide to merge labels.
}

TARGET_COL = "Rupture"
GROUP_COL = "Center"
ID_COL = "Patient ID"

# Exact spreadsheet columns
# Full model includes antithrombotic medication and hereditary disease by default.
ALL_FEATURES = [
    "Age_years",
    "Male",
    "Smoking",
    "CVD",
    "HTN",
    "DM",
    "Fam_Hx",
    "Eloquent",
    "Deep",
    "Nidus_size",
    "Compacted",
    "Location_frontal",
    "Location_temporal",
    "Location_parietal",
    "Location_occipital",
    "Location_cerebellar",
    "Location_thalamic",
    "Location_brainstem",
    "Location_corpus_callosum",
    "Location_insular",
    "Location_basal_ganglia",
    "Feeder_diameter_gt1mm",
    "Feeders_from_vertebrobasilar",
    "Location_DV_superficial",
    "Location_DV_deep",
    "Location_DV_both",
    "Venous_stenosis",
    "Nidal_aneurysm",
    "Aneurysm_location_nidal",
    "Aneurysm_location_prenidal",
    "Aneurysm_location_venous",
    "Race_white",
    "Race_african_american",
    "Race_hispanic",
    "Race_native_american",
    "Race_asian",
    "Race_other",
    "Side_right",
    "Side_left",
    "Side_midline",
    "Side_bilateral",
    "Feeders_multiple",
    "Underlying_hereditary_disease",
    "Antithrombotic_medication",
    "Draining_dural_venous_sinuses",
    "Draining_deep_cerebral_veins",
    "Draining_superficial_cortical_veins",
    "DV_multiple",
    "Feeders_from_middle_cerebral_artery",
    "Feeders_from_anterior_cerebral_artery",
    "Feeders_from_posterior_cerebral_artery",
    "Feeders_from_posterior_inferior_cerebellar_artery",
    "Feeders_from_superior_cerebellar_artery",
    "Feeders_from_other",
]

# Exact 5-variable R2eD approximation from available columns.
# Note: "exclusive deep location" and "exclusive deep venous drainage" require approximation
# because the spreadsheet does not include a single pre-made R2eD field.
R2ED_COMPONENTS = [
    "R2ED_nonwhite",
    "R2ED_exclusive_deep_location",
    "R2ED_small_size_lt3cm",
    "R2ED_exclusive_deep_venous_drainage",
    "R2ED_monoarterial",
]


# ============================================================
# 2. DATA LOADING / CLEANING
# ============================================================
def load_data(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)

    # Harmonize centers only if explicitly configured.
    if CENTER_MAP:
        df[GROUP_COL] = df[GROUP_COL].replace(CENTER_MAP)

    # Remove rows without target or center.
    df = df[df[TARGET_COL].notna()].copy()
    df = df[df[GROUP_COL].notna()].copy()

    # Ensure binary outcome is 0/1 int.
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Coerce numeric predictors to numeric where possible.
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# 3. R2eD ENGINEERING (DIRECT DATASET MAPPING)
# ============================================================
def engineer_r2ed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Nonwhite race = any race category other than white.
    nonwhite_cols = [
        "Race_african_american",
        "Race_hispanic",
        "Race_native_american",
        "Race_asian",
        "Race_other",
    ]
    out["R2ED_nonwhite"] = out[nonwhite_cols].fillna(0).max(axis=1)

    # Exclusive deep location is directly encoded in this dataset as `Deep`.
    out["R2ED_exclusive_deep_location"] = out["Deep"].astype(float)
    out.loc[out["Deep"].isna(), "R2ED_exclusive_deep_location"] = np.nan

    # Size < 3 cm
    out["R2ED_small_size_lt3cm"] = (out["Nidus_size"] < 3).astype(float)
    out.loc[out["Nidus_size"].isna(), "R2ED_small_size_lt3cm"] = np.nan

    # Exclusive deep venous drainage is directly encoded in this dataset as `Location_DV_deep`.
    out["R2ED_exclusive_deep_venous_drainage"] = out["Location_DV_deep"].astype(float)
    out.loc[out["Location_DV_deep"].isna(), "R2ED_exclusive_deep_venous_drainage"] = np.nan

    # Monoarterial feeding = inverse of multiple feeders.
    out["R2ED_monoarterial"] = (1 - out["Feeders_multiple"].fillna(0)).astype(float)
    out.loc[out["Feeders_multiple"].isna(), "R2ED_monoarterial"] = np.nan

    # Original score weighting: race=2 points; all others=1 point.
    out["R2ED_score"] = (
        2 * out["R2ED_nonwhite"].fillna(0)
        + out["R2ED_exclusive_deep_location"].fillna(0)
        + out["R2ED_small_size_lt3cm"].fillna(0)
        + out["R2ED_exclusive_deep_venous_drainage"].fillna(0)
        + out["R2ED_monoarterial"].fillna(0)
    )

    return out


# ============================================================
# 4. PREPROCESSORS / MODELS
# ============================================================
def make_elastic_net_pipeline(feature_names: List[str]) -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_names)],
        remainder="drop",
    )

    # Saga supports elastic net for binary classification.
    clf = LogisticRegressionCV(
        Cs=20,
        cv=4,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        scoring="roc_auc",
        max_iter=10000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", clf),
    ])


def make_r2ed_logistic_pipeline(feature_names: List[str]) -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_names)],
        remainder="drop",
    )

    clf = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", clf),
    ])


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    feature_names: List[str],
) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
        verbosity=-1,
    )

    param_dist = {
        "n_estimators": [200, 300, 500, 700],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 3, 5, 7, 9],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
    }

    inner_cv = StratifiedGroupKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="roc_auc",
        cv=inner_cv.split(X_train[feature_names], y_train, groups_train),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train[feature_names], y_train)
    return search.best_estimator_


# ============================================================
# 5. METRICS / UTILITIES
# ============================================================
def threshold_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Threshold": threshold,
    }


def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)

    # Calibration intercept: logistic model with slope fixed to 1 is ideal, but here simple approximation.
    # Practical alternative: regress y on logit(p).
    cal_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    cal_model.fit(logit_p, y_true)
    slope = float(cal_model.coef_[0][0])
    intercept = float(cal_model.intercept_[0])
    return {"CalibrationSlope": slope, "CalibrationIntercept": intercept}


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_prob[idx]))
    vals = np.array(vals)
    return np.nanpercentile(vals, [2.5, 97.5])


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    n = len(y_true)
    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        records.append({"threshold": t, "net_benefit": nb})
    return pd.DataFrame(records)


# ============================================================
# 6. CROSS-VALIDATED BENCHMARKING
# ============================================================
def run_benchmark(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    y = df[TARGET_COL].astype(int)
    groups = df[GROUP_COL].astype(str)

    outer_cv = StratifiedGroupKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Out-of-fold predictions
    oof = {
        "R2ED_raw": np.full(len(df), np.nan),
        "R2ED_score_recalibrated": np.full(len(df), np.nan),
        "R2ED_logistic": np.full(len(df), np.nan),
        "ElasticNet_full": np.full(len(df), np.nan),
        "LightGBM_full": np.full(len(df), np.nan),
    }

    fitted_models = {"LightGBM_full": []}

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df, y, groups), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        X_train_full = train_df[ALL_FEATURES]
        X_test_full = test_df[ALL_FEATURES]
        y_train = train_df[TARGET_COL].astype(int)
        y_test = test_df[TARGET_COL].astype(int)
        groups_train = train_df[GROUP_COL].astype(str)

        # Model 0: raw R2eD point score as an uncalibrated bedside comparator.
        # For unified plotting/metric code, scale the original 0-6 integer score to 0-1.
        # This preserves rank-order discrimination exactly, while making Brier/calibration outputs
        # interpretable as a crude normalized risk indicator rather than a fitted probability model.
        train_score = train_df[["R2ED_score"]].copy()
        test_score = test_df[["R2ED_score"]].copy()
        oof["R2ED_raw"][test_idx] = (test_score["R2ED_score"].astype(float) / 6.0).values

        # Model 1: fold-specific recalibration of the raw R2eD point score.
        r2ed_score_cal = make_r2ed_logistic_pipeline(["R2ED_score"])
        r2ed_score_cal.fit(train_score, y_train)
        oof["R2ED_score_recalibrated"][test_idx] = r2ed_score_cal.predict_proba(test_score)[:, 1]

        # Model 2: logistic regression on the 5 R2eD components.
        r2ed_logit = make_r2ed_logistic_pipeline(R2ED_COMPONENTS)
        r2ed_logit.fit(train_df[R2ED_COMPONENTS], y_train)
        oof["R2ED_logistic"][test_idx] = r2ed_logit.predict_proba(test_df[R2ED_COMPONENTS])[:, 1]

        # Model 3: full elastic-net logistic regression.
        enet = make_elastic_net_pipeline(ALL_FEATURES)
        enet.fit(X_train_full, y_train)
        oof["ElasticNet_full"][test_idx] = enet.predict_proba(X_test_full)[:, 1]

        # Model 4: full LightGBM with inner tuning.
        lgbm = tune_lightgbm(train_df, y_train, groups_train, ALL_FEATURES)
        lgbm.fit(X_train_full, y_train)
        oof["LightGBM_full"][test_idx] = lgbm.predict_proba(X_test_full)[:, 1]
        fitted_models["LightGBM_full"].append(lgbm)

        print(f"Finished outer fold {fold}/{N_OUTER_SPLITS}")

    # Summary metrics using one global threshold selected from OOF predictions.
    summary_rows = []
    for model_name, probs in oof.items():
        thr = threshold_by_youden(y.values, probs)
        metrics = compute_metrics(y.values, probs, thr)
        metrics.update(calibration_stats(y.values, probs))
        metrics["Model"] = model_name

        # Add bootstrap CIs for core metrics.
        auroc_ci = bootstrap_ci(y.values, probs, roc_auc_score)
        auprc_ci = bootstrap_ci(y.values, probs, average_precision_score)
        brier_ci = bootstrap_ci(y.values, probs, brier_score_loss)
        metrics["AUROC_95CI"] = f"{auroc_ci[0]:.3f}-{auroc_ci[1]:.3f}"
        metrics["AUPRC_95CI"] = f"{auprc_ci[0]:.3f}-{auprc_ci[1]:.3f}"
        metrics["Brier_95CI"] = f"{brier_ci[0]:.3f}-{brier_ci[1]:.3f}"
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows).sort_values("AUROC", ascending=False)
    return summary, oof, fitted_models


# ============================================================
# 7. PLOTS / EXPORTS
# ============================================================
def plot_roc_curves(y_true: np.ndarray, oof: Dict[str, np.ndarray], outpath: Path):
    plt.figure(figsize=(8, 6))
    for name, probs in oof.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("Out-of-fold ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_pr_curves(y_true: np.ndarray, oof: Dict[str, np.ndarray], outpath: Path):
    plt.figure(figsize=(8, 6))
    for name, probs in oof.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        auprc = average_precision_score(y_true, probs)
        plt.plot(recall, precision, label=f"{name} (AUPRC={auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Out-of-fold precision-recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_calibration(y_true: np.ndarray, oof: Dict[str, np.ndarray], outpath: Path):
    plt.figure(figsize=(8, 6))
    for name, probs in oof.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed event rate")
    plt.title("Calibration plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_decision_curve(y_true: np.ndarray, oof: Dict[str, np.ndarray], outpath: Path):
    thresholds = np.linspace(0.05, 0.80, 76)
    plt.figure(figsize=(8, 6))
    for name, probs in oof.items():
        nb_df = net_benefit(y_true, probs, thresholds)
        plt.plot(nb_df["threshold"], nb_df["net_benefit"], label=name)

    prevalence = np.mean(y_true)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)
    plt.plot(thresholds, treat_all, linestyle="--", label="Treat all")
    plt.plot(thresholds, treat_none, linestyle=":", label="Treat none")

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision-curve analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def export_missingness(df: pd.DataFrame, outpath: Path):
    miss = df[[TARGET_COL, GROUP_COL] + ALL_FEATURES].isna().mean().sort_values(ascending=False)
    miss_df = (miss * 100).round(2).rename("PercentMissing").reset_index().rename(columns={"index": "Variable"})
    miss_df.to_csv(outpath, index=False)


# ============================================================
# 8. FINAL FIT + SHAP-READY EXPORT
# ============================================================
def fit_final_lightgbm(df: pd.DataFrame) -> LGBMClassifier:
    y = df[TARGET_COL].astype(int)
    groups = df[GROUP_COL].astype(str)
    model = tune_lightgbm(df, y, groups, ALL_FEATURES)
    model.fit(df[ALL_FEATURES], y)
    return model


def export_feature_importance(model: LGBMClassifier, outpath: Path):
    imp = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    imp.to_csv(outpath, index=False)


# ============================================================
# 9. MAIN
# ============================================================
def main():
    print("Loading data...")
    df = load_data(EXCEL_PATH)
    df = engineer_r2ed(df)

    print(f"Rows after target filtering: {len(df)}")
    print(f"Ruptured: {df[TARGET_COL].sum()} | Unruptured: {(1 - df[TARGET_COL]).sum()}")
    print(f"Unique centers: {df[GROUP_COL].nunique()}")

    export_missingness(df, OUTPUT_DIR / "missingness_table.csv")

    print("Running grouped cross-validated benchmarking...")
    summary, oof, _ = run_benchmark(df)
    summary.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)

    y_true = df[TARGET_COL].astype(int).values
    oof_df = pd.DataFrame({"Patient_ID": df[ID_COL].values, "Center": df[GROUP_COL].values, "Rupture": y_true})
    for k, v in oof.items():
        oof_df[k] = v
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    plot_roc_curves(y_true, oof, OUTPUT_DIR / "roc_curves.png")
    plot_pr_curves(y_true, oof, OUTPUT_DIR / "pr_curves.png")
    plot_calibration(y_true, oof, OUTPUT_DIR / "calibration_plot.png")
    plot_decision_curve(y_true, oof, OUTPUT_DIR / "decision_curve.png")

    print("Fitting final LightGBM model on full cohort...")
    final_lgbm = fit_final_lightgbm(df)
    export_feature_importance(final_lgbm, OUTPUT_DIR / "lightgbm_feature_importance.csv")

    print("Done. Outputs saved to:", OUTPUT_DIR.resolve())
    print(summary[["Model", "AUROC", "AUPRC", "Brier", "Sensitivity", "Specificity", "CalibrationSlope", "CalibrationIntercept"]])


if __name__ == "__main__":
    main()
