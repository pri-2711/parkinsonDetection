# Pipeline results

All classification estimates use nested five-fold subject-disjoint cross-validation. SVM/RF probabilities are sigmoid-calibrated using group-disjoint folds. Ensemble weights are chosen within each outer training partition; reported ensemble scores use outer held-out folds. The deployable ensemble weight is selected from full-cohort grouped OOF predictions and is therefore not a final independent test estimate.

## Modality classification

| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| speech | svm | 0.796 | 0.826 | 0.750 | 0.786 | 0.844 |
| speech | random_forest | 0.775 | 0.795 | 0.742 | 0.767 | 0.830 |
| speech | ensemble | 0.783 | 0.809 | 0.742 | 0.774 | 0.835 |
| handwriting | svm | 0.891 | 0.859 | 1.000 | 0.924 | 0.812 |
| handwriting | random_forest | 0.909 | 0.889 | 0.986 | 0.935 | 0.810 |
| handwriting | ensemble | 0.891 | 0.867 | 0.986 | 0.923 | 0.805 |
| tremor | svm | 0.823 | 0.667 | 0.467 | 0.549 | 0.786 |
| tremor | random_forest | 0.815 | 0.688 | 0.367 | 0.478 | 0.857 |
| tremor | ensemble | 0.815 | 0.667 | 0.400 | 0.500 | 0.850 |

## Classification comparison with prior reported scores

Current results use nested CV and calibrated probabilities. Earlier scores are included as historical reference; small changes in preprocessing and evaluation nesting mean they should not be interpreted as a controlled model-only comparison.

| Dataset | Model | Prior F1 | Current F1 | Prior ROC-AUC | Current ROC-AUC |
|---|---|---:|---:|---:|---:|
| speech | svm | 0.791 | 0.786 | 0.850 | 0.844 |
| speech | random_forest | 0.767 | 0.767 | 0.831 | 0.830 |
| handwriting | svm | 0.896 | 0.924 | 0.866 | 0.812 |
| handwriting | random_forest | 0.980 | 0.935 | 0.958 | 0.810 |
| tremor | svm | 0.560 | 0.549 | 0.819 | 0.786 |
| tremor | random_forest | 0.429 | 0.478 | 0.823 | 0.857 |

## Telemonitoring regression

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| ridge | 7.656 | 9.089 | -0.250 |
| random_forest | 8.096 | 9.841 | -0.466 |
| group_mean_baseline | 7.187 | 8.316 | -0.047 |

## Selected deployable ensemble weights

Weights were selected from full-cohort grouped OOF predictions. Use nested outer-CV scores above to estimate weight selection performance.

| Modality | SVM weight | RF weight |
|---|---:|---:|
| speech | 0.9 | 0.1 |
| handwriting | 0.8 | 0.2 |
| tremor | 0.2 | 0.8 |

## Data, features, and split strategy

| Dataset | Cohort | Features used | Split group |
|---|---|---|---|
| Speech | 240 recordings, 80 participants | 45 numeric acoustic features; ID, recording, and status excluded | `id` |
| Handwriting | 110 tests, 40 participants (15 control, 25 PD) | 16 aggregates: point count, path length, velocity, acceleration, jerk, pressure, grip angle, duration, pen-up fraction, spatial spread, bounding box | `participant_id`, joined across exact duplicate feature signatures |
| Tremor | 130 participants (100 control, 30 PD) | 7 numeric UPDRS items from items 20/21; missing values imputed within training folds | `participant_code` |
| Telemonitoring | 5,875 recordings, 42 participants | 19 predictors: age, sex, test time, jitter/shimmer, NHR/HNR, RPDE, DFA, PPE; subject ID and total UPDRS excluded | `subject_number` |

All classification estimates use nested five-fold `StratifiedGroupKFold`; regression uses five-fold `GroupKFold`. Imputation, scaling, calibration, and weight choice are training-fold-only. Handwriting feature extraction follows source point order. The 7-row control signature and 2-row PD signature shared across participants are each held in one fold group.

## Regression baseline comparison

| Model | Prior MAE / RMSE / R2 | Current MAE / RMSE / R2 |
|---|---:|---:|
| Ridge | 7.071 / 8.298 / -0.042 | 7.656 / 9.089 / -0.250 |
| Random Forest | 7.486 / 9.192 / -0.279 | 8.096 / 9.841 / -0.466 |

## Interpretation and limitations

Telemonitoring does not currently generalize usefully to unseen participants because R2 is negative. The handwriting audit found exact feature signatures shared across participants; handwriting folds keep matching signatures together to avoid cross-fold contamination. Scores remain exploratory because there are only 40 participants and external validation is absent. No three-modality fusion was trained because these cohorts are unmatched. See per-dataset JSON results, feature summaries, and the handwriting leakage audit for details.
