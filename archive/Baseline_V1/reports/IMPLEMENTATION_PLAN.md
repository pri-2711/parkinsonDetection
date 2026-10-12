# Pipeline implementation plan

## Repository inspection

- `main.py` reruns cleaning, handwriting aggregation, and then three classifiers per modality plus four telemonitoring regressors.
- Current classification/regression metrics come from five-fold subject-grouped out-of-fold predictions. Preprocessing is fitted per fold, and final pipelines are refitted on all rows.
- Classification currently uses `SVC(probability=True)`, random forest, and gradient boosting. There is no explicit probability calibration or ensemble weight selection.
- Speech and telemonitoring cleaning currently impute medians before splitting, which leaks information across folds. These imputations will be left to training-fold pipelines.
- Handwriting is aggregated into participant/test rows; folds group on participant. Ordered source points and participant/test keys are retained. Very high scores require an audit for duplicate/near duplicate rows, label consistency, participant overlap, target-like features, and per-participant test count.
- Telemonitoring excludes `total_updrs` from its motor-UPDRS predictors, but other repeated-session and subject-identity effects must be discussed in the report.
- `model_pipeline.py` currently describes average three-modality fusion despite the absence of matched subjects; it will be made explicit that fusion is future work.

## Implementation

1. Add a small `src/` package for configuration, validation, feature summaries, split checks, fold-local model pipelines, nested group evaluation, reports, and artifacts.
2. Preserve raw datasets. Keep cleaned tables and feature extraction deterministic; remove dataset-wide imputation from cleaning and perform it inside sklearn pipelines.
3. Validate schemas, labels, missing values, group counts, and class counts before fitting. Use five-fold `StratifiedGroupKFold` for classification where each test fold can contain both labels; otherwise fail with a clear reason. Use five-fold `GroupKFold` for motor-UPDRS regression.
4. Use only calibrated RBF SVM and calibrated Random Forest for classification. Compare them using out-of-fold predictions and choose blend weights from nested group-aware validation only. Selection score: equal-weight mean of ROC-AUC and positive-class F1 at threshold 0.5; resolve ties by recall, then closeness to a 0.5 SVM weight. Report recall explicitly. Outer folds estimate performance.
5. Use Ridge and Random Forest as regression baselines; keep every preprocessor fold-local. Compare against a group-held-out training-mean predictor and current checked-in baseline numbers.
6. Save model pipelines, selected weights, feature names/metadata, config, metrics, validation summaries, and human-readable reports in `models/artifacts/` and `reports/`.
7. Run the requested complete entry point, inspect generated reports and artifacts, and record any unsupported conclusions or runtime constraints.

## Methodological limits to document

- Grouped cross-validation estimates performance on unseen people within these cohorts; small cohort sizes make scores uncertain.
- The tremor label is derived from the participant code in the existing dataset convention; this needs validation against source documentation.
- The handwriting cohort has only 40 participants. Any extremely high result remains exploratory pending external validation.
- The telemonitoring task predicts motor UPDRS in unseen subjects, not future severity for known subjects. The small 42-subject cohort can make this hard, and negative R² means no useful generalization was demonstrated.
- No multimodal fusion model will be trained because these three classification datasets do not identify a matched common cohort.
