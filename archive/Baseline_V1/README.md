# Parkinson's Detection and Severity Estimation

This repository contains two separate research tasks: subject-level Parkinson's classification from tremor, speech, and handwriting, and motor-UPDRS regression from telemonitoring recordings. No multimodal classifier is trained because the classification cohorts do not share matched participants.

## Reproducible pipeline

Run from the repository root:

```powershell
python scripts/run_all.py
```

The entry point regenerates cleaned tables and handwriting participant/test features from the raw files (raw files are never edited), validates and profiles the data, performs grouped evaluation, fits final pipelines, and writes reports/artifacts. `python main.py` remains a compatibility entry point.

## Design

- Speech groups by `id`; handwriting groups by `participant_id`; tremor groups by `participant_code`; telemonitoring groups by `subject_number`.
- Classification uses five-fold `StratifiedGroupKFold`; regression uses five-fold `GroupKFold`. Preprocessing/imputation and sigmoid probability calibration are fitted within training folds.
- Classification compares calibrated RBF SVM and calibrated Random Forest. Nested grouped validation chooses an SVM/RF blend weight from 0.0 through 1.0. The selection score is the mean of ROC-AUC and F1 at probability threshold 0.5; ties use recall then closeness to equal weights. Outer folds estimate performance.
- Severity regression compares Ridge and Random Forest against a training-fold mean baseline. `total_updrs` and subject ID are excluded from predictors.
- Handwriting features are compact participant/test aggregates. Source trajectory order and participant/test boundaries are preserved; model folds group by participant and omit participant/test identifiers.
- The tremor target follows the current source convention (`participant_code` beginning with `PD`); predictors are limited to UPDRS items 20 and 21.

## Layout

```text
src/                    Pipeline configuration, modeling, grouped evaluation
scripts/run_all.py      Single complete pipeline entry point
cleaningPipeline/       Deterministic raw-to-clean data transforms
featureEngineering/     Handwriting trajectory aggregation
raw_dataset/            Original source data (read only)
cleaned_dataset/        Cleaned source tables
models/artifacts/       Joblib model bundles, out-of-fold predictions, run config
reports/data_profile/   Dataset profiles
reports/feature_analysis/ Feature summaries and correlations
reports/leakage_audit/  Handwriting leakage audit
reports/model_results/ Per-task metrics and weight comparisons
reports/FINAL_REPORT.md Pipeline findings and interpretation
reports/IMPLEMENTATION_PLAN.md Initial inspection and implementation plan
```

Install Python dependencies with `pip install -r requirements.txt`. See generated reports for results and the limitations of the small, unmatched cohorts. A model with negative test-fold R² is not useful for unseen-subject severity prediction; unusually high handwriting scores still require external validation.
