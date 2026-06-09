# AVM Hemorrhagic-Presentation Prediction — ML Pipeline

A grouped, cross-validated benchmarking pipeline for predicting **hemorrhagic (ruptured) presentation of brain arteriovenous malformations at diagnosis** from clinical and angioarchitectural features. A hyperparameter-tuned LightGBM model is benchmarked against the published **R2eD AVM score**, a recalibrated R2eD score, an R2eD-component logistic model, and a full elastic-net logistic regression.

> **Outcome note:** the target is *hemorrhagic presentation at diagnosis* (a cross-sectional label), **not** prospective/annual rupture risk.

## ⚠️ Data and privacy

This repository contains **code only**. The input spreadsheet holds individual patient records and is **not included** and must not be committed.

- Supply your own data as `MISTA_Analysis.xlsx` (or edit `EXCEL_PATH` in the script) using the schema below.
- The generated `oof_predictions.csv` contains a `Patient ID` column; it and the entire `avm_ml_outputs/` directory are git-ignored. Do not commit them.

## What it does

1. Loads the cohort, drops rows missing the outcome or center, coerces predictors to numeric.
2. Derives the five R2eD components and the 0–6 R2eD point score (race weighted 2 points).
3. Runs a nested, site-grouped cross-validated benchmark of five models and produces out-of-fold (OOF) predictions.
4. Computes discrimination, calibration, and net-benefit metrics with bootstrap confidence intervals.
5. Fits a final LightGBM model on the full cohort and exports its feature importances.

## Models compared

| Name | Description |
|---|---|
| `R2ED_raw` | Published R2eD 0–6 point score, normalised to 0–1 (rank-preserving comparator) |
| `R2ED_score_recalibrated` | Logistic recalibration of the R2eD point score |
| `R2ED_logistic` | Unpenalised logistic regression on the 5 R2eD components |
| `ElasticNet_full` | Elastic-net logistic regression (`LogisticRegressionCV`, saga) on all features |
| `LightGBM_full` | Gradient-boosted trees with inner-loop randomized hyperparameter search |

## Methods

- **Validation:** nested grouped CV with patients grouped by `Center`. Outer = `StratifiedGroupKFold` (5 splits); inner = `StratifiedGroupKFold` (4 splits) for tuning. Grouping prevents same-site leakage across folds.
- **Tuning:** `RandomizedSearchCV` (30 iterations, ROC-AUC) over LightGBM depth, leaves, learning rate, estimators, subsampling, and L1/L2 regularisation.
- **Missing data:** elastic-net and R2eD-logistic use imputation + standardisation inside the pipeline; LightGBM uses its native missing-value handling (no imputation), so blank inputs are first-class.
- **Metrics:** AUROC, AUPRC, Brier, and sensitivity/specificity/PPV/NPV at a single global Youden-optimal threshold derived from OOF predictions; calibration slope and intercept (regression of outcome on the logit of predicted risk); bootstrap 95% CIs (1,000 resamples).
- **Clinical utility:** decision-curve net benefit across threshold probabilities 0.05–0.80.
- **Seed:** `RANDOM_STATE = 42` throughout.

## Data schema

The spreadsheet is expected to contain these columns.

- **Outcome / keys:** `Rupture` (0/1), `Center` (site label), `Patient ID`
- **Demographics & comorbidities:** `Age_years`, `Male`, `Smoking`, `CVD`, `HTN`, `DM`, `Fam_Hx`, `Antithrombotic_medication`, `Underlying_hereditary_disease`
- **Nidus:** `Nidus_size` (cm), `Compacted`, `Eloquent`, `Deep`
- **Location (one-hot):** `Location_frontal`, `Location_temporal`, `Location_parietal`, `Location_occipital`, `Location_cerebellar`, `Location_thalamic`, `Location_brainstem`, `Location_corpus_callosum`, `Location_insular`, `Location_basal_ganglia`
- **Side (one-hot):** `Side_right`, `Side_left`, `Side_midline`, `Side_bilateral`
- **Arterial supply:** `Feeders_multiple`, `Feeder_diameter_gt1mm`, `Feeders_from_vertebrobasilar`, `Feeders_from_middle_cerebral_artery`, `Feeders_from_anterior_cerebral_artery`, `Feeders_from_posterior_cerebral_artery`, `Feeders_from_posterior_inferior_cerebellar_artery`, `Feeders_from_superior_cerebellar_artery`, `Feeders_from_other`
- **Venous drainage:** `Location_DV_superficial`, `Location_DV_deep`, `Location_DV_both`, `Draining_dural_venous_sinuses`, `Draining_deep_cerebral_veins`, `Draining_superficial_cortical_veins`, `DV_multiple`, `Venous_stenosis`
- **Aneurysms:** `Nidal_aneurysm`, `Aneurysm_location_nidal`, `Aneurysm_location_prenidal`, `Aneurysm_location_venous`
- **Race (one-hot):** `Race_white`, `Race_african_american`, `Race_hispanic`, `Race_native_american`, `Race_asian`, `Race_other`

Binary fields are 0/1; blanks are treated as missing. `CENTER_MAP` can optionally harmonise site labels.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Usage

Place `MISTA_Analysis.xlsx` next to the script (or edit `EXCEL_PATH`), then:

```bash
python avm_hemorrhage_pipeline.py
```

## Outputs

Written to `avm_ml_outputs/`:

- `benchmark_summary.csv` — per-model AUROC / AUPRC / Brier / operating-point stats / calibration, with 95% CIs
- `roc_curves.png`, `pr_curves.png`, `calibration_plot.png`, `decision_curve.png`
- `lightgbm_feature_importance.csv` — final-model importances
- `missingness_table.csv` — per-variable percent missing
- `oof_predictions.csv` — out-of-fold probabilities per model *(contains `Patient ID`; git-ignored)*

## Reproducibility

All splits, the search, and the models use `RANDOM_STATE = 42`. Note that LightGBM tree construction can depend on platform and thread count, so exact tree structure and feature importances may vary slightly between environments even with a fixed seed; headline discrimination and calibration are stable.

## Reference

Feghali J, Yang W, Xu R, et al. *The R2eD AVM Score: A Novel Predictive Tool for Arteriovenous Malformation Presentation With Hemorrhage.* Stroke. 2019;50(7):1703–1710. doi:10.1161/STROKEAHA.119.025054

## License

No license file is included yet — add one before making the repository public if you intend to permit reuse (MIT is a common permissive choice for research code).
