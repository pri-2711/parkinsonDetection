# Regularized SVM and Random Forest classification experiment

This experiment is isolated under `experiments/regularized_svm_rf`; Baseline V1 remains archived unchanged.

## Protocol

The archived cleaned inputs/features and archived baseline OOF predictions were used. Five-fold `StratifiedGroupKFold` (seed 42) defines the same outer participant partitions as Baseline V1. Each outer training partition uses four-fold grouped inner grid search by ROC-AUC; sigmoid calibration uses three grouped folds after hyperparameter selection. Median imputation and SVM scaling are in the estimator pipeline and refit inside every inner/calibration training fold. Thresholded metrics use 0.5.

Fold 1 in each modality is reserved for validation-only AUC-based fusion-weight estimation. It is excluded from the reported baseline/tuned classification metrics; comparisons therefore use the same four outer test folds (folds 2–5). The fusion-validation fold is not an independent multimodal patient cohort.

## Baseline V1 vs tuned results

Baseline probabilities come from the archived OOF file; tuned probabilities were generated on the identical row assignments and outer group folds. The nested-selected row chooses SVM or RF within each training fold using inner-CV ROC-AUC.

| Modality | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| speech | Baseline SVM | 0.7656 | 0.7931 | 0.7188 | 0.7541 | 0.8171 |
| speech | Tuned SVM | 0.7604 | 0.7778 | 0.7292 | 0.7527 | 0.8238 |
| speech | Baseline RF | 0.7552 | 0.7692 | 0.7292 | 0.7487 | 0.8111 |
| speech | Tuned RF | 0.7604 | 0.7778 | 0.7292 | 0.7527 | 0.8130 |
| speech | Nested selected SVM/RF | 0.7604 | 0.7778 | 0.7292 | 0.7527 | 0.8238 |
| handwriting | Baseline SVM | 0.9733 | 0.9672 | 1.0000 | 0.9833 | 0.9523 |
| handwriting | Tuned SVM | 0.9467 | 0.9661 | 0.9661 | 0.9661 | 0.9322 |
| handwriting | Baseline RF | 0.9600 | 0.9667 | 0.9831 | 0.9748 | 0.9089 |
| handwriting | Tuned RF | 0.9600 | 0.9667 | 0.9831 | 0.9748 | 0.9460 |
| handwriting | Nested selected SVM/RF | 0.9467 | 0.9661 | 0.9661 | 0.9661 | 0.9460 |
| tremor | Baseline SVM | 0.8365 | 0.6842 | 0.5417 | 0.6047 | 0.8305 |
| tremor | Tuned SVM | 0.8269 | 0.6667 | 0.5000 | 0.5714 | 0.8997 |
| tremor | Baseline RF | 0.8077 | 0.6667 | 0.3333 | 0.4444 | 0.8505 |
| tremor | Tuned RF | 0.8173 | 0.6667 | 0.4167 | 0.5128 | 0.8427 |
| tremor | Nested selected SVM/RF | 0.8269 | 0.6667 | 0.5000 | 0.5714 | 0.8997 |

## Fold-level variability

Values below use folds 2–5 only (mean ± sample standard deviation). Fold 1 is reserved for fusion validation.

| Modality | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| speech | Baseline SVM | 0.7656 ± 0.0356 | 0.7946 ± 0.0506 | 0.7188 ± 0.0208 | 0.7545 ± 0.0330 | 0.8194 ± 0.0294 |
| speech | Tuned SVM | 0.7604 ± 0.0269 | 0.7784 ± 0.0271 | 0.7292 ± 0.0538 | 0.7521 ± 0.0326 | 0.8286 ± 0.0321 |
| speech | Baseline RF | 0.7552 ± 0.0199 | 0.7712 ± 0.0375 | 0.7292 ± 0.0241 | 0.7488 ± 0.0145 | 0.8151 ± 0.0384 |
| speech | Tuned RF | 0.7604 ± 0.0269 | 0.7801 ± 0.0447 | 0.7292 ± 0.0241 | 0.7530 ± 0.0217 | 0.8207 ± 0.0373 |
| speech | Nested selected SVM/RF | 0.7604 ± 0.0269 | 0.7784 ± 0.0271 | 0.7292 ± 0.0538 | 0.7521 ± 0.0326 | 0.8286 ± 0.0321 |
| handwriting | Baseline SVM | 0.9722 ± 0.0556 | 0.9688 ± 0.0625 | 1.0000 ± 0.0000 | 0.9833 ± 0.0333 | 0.9598 ± 0.0804 |
| handwriting | Tuned SVM | 0.9459 ± 0.0625 | 0.9688 ± 0.0625 | 0.9667 ± 0.0667 | 0.9655 ± 0.0399 | 0.9688 ± 0.0625 |
| handwriting | Baseline RF | 0.9591 ± 0.0530 | 0.9688 ± 0.0625 | 0.9833 ± 0.0333 | 0.9747 ± 0.0320 | 0.9375 ± 0.1250 |
| handwriting | Tuned RF | 0.9591 ± 0.0530 | 0.9688 ± 0.0625 | 0.9833 ± 0.0333 | 0.9747 ± 0.0320 | 0.9821 ± 0.0357 |
| handwriting | Nested selected SVM/RF | 0.9459 ± 0.0625 | 0.9688 ± 0.0625 | 0.9667 ± 0.0667 | 0.9655 ± 0.0399 | 0.9821 ± 0.0357 |
| tremor | Baseline SVM | 0.8365 ± 0.0368 | 0.6833 ± 0.0839 | 0.5417 ± 0.1596 | 0.5960 ± 0.1262 | 0.8531 ± 0.0629 |
| tremor | Tuned SVM | 0.8269 ± 0.0222 | 0.6708 ± 0.0614 | 0.5000 ± 0.1361 | 0.5641 ± 0.0939 | 0.9073 ± 0.0216 |
| tremor | Baseline RF | 0.8077 ± 0.0314 | 0.6458 ± 0.1049 | 0.3333 ± 0.1361 | 0.4347 ± 0.1433 | 0.8687 ± 0.0269 |
| tremor | Tuned RF | 0.8173 ± 0.0192 | 0.6667 ± 0.0000 | 0.4167 ± 0.1667 | 0.5000 ± 0.1111 | 0.8667 ± 0.0295 |
| tremor | Nested selected SVM/RF | 0.8269 ± 0.0222 | 0.6708 ± 0.0614 | 0.5000 ± 0.1361 | 0.5641 ± 0.0939 | 0.9073 ± 0.0216 |

## Selected deployment models and parameters

- **speech: svm**. Full-cohort grouped-CV ROC-AUC: 0.8742. Parameters: `{'model__C': 1.0, 'model__class_weight': None, 'model__gamma': 0.01}`.
- **handwriting: svm**. Full-cohort grouped-CV ROC-AUC: 0.9663. Parameters: `{'model__C': 0.1, 'model__class_weight': None, 'model__gamma': 0.01}`.
- **tremor: svm**. Full-cohort grouped-CV ROC-AUC: 0.8563. Parameters: `{'model__C': 1.0, 'model__class_weight': None, 'model__gamma': 0.1}`.

## Fusion validation

The existing AUC-normalized late-fusion function preserves probability averaging and renormalizes weights when modalities are missing. Its Baseline V1 weights were derived from all outer OOF `actual` labels and therefore do not meet the stricter requirement that evaluation-fold labels not determine weights. Baseline V1 is preserved unchanged. The isolated experiment estimates AUC-normalized weights from fold 1 only, which was excluded from reported per-modality test metrics; weights are recorded in `fusion_validation_weights.json`. This is a validation-only modality reliability check, not patient-level multimodal validation.

The Baseline V1 AUC-normalized weights are reproduced below for audit. Those AUCs came from the complete modality OOF files, including every outer held-out fold, so their labels were part of the fusion-weight calculation. The experiment-local weights use only reserved fold 1; folds 2–5 do not influence those weights.

| Modality | Baseline V1 outer-OOF AUC | Baseline V1 normalized weight | Reserved validation AUC | Validation-only normalized weight |
|---|---:|---:|---:|---:|
| speech | 0.835347 | 0.335332 | 0.940972 | 0.350142 |
| handwriting | 0.805257 | 0.323253 | 0.904762 | 0.336668 |
| tremor | 0.850500 | 0.341415 | 0.841667 | 0.313190 |

## External validation

No suitable independent local test cohort was identified. The extra handwriting directory contains Parkinson-only files and lacks controls in the engineered tabular feature schema; the spiral image directories have no aligned participant labels/features; speech and tremor each contain only the cohorts represented in the primary inputs. External validation remains pending. No portal or external data source was searched.

## Reproduction

From the repository root:

```powershell
python experiments/regularized_svm_rf/run_experiments.py
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/regularized_svm_rf -p 'test_*.py' -v
python archive/Baseline_V1/verify_archive.py
```

The run writes only below `experiments/regularized_svm_rf/results`. The archived model/data/config and manifest are read-only inputs. See `run_metadata.json` for package versions, seeds, and data hashes.

## Limitations

The first fold is reserved, so reported classification metrics cover four outer folds rather than the full five-fold estimate. Fold-to-fold variability is high for small cohorts. Full-cohort deployment hyperparameters are selected via grouped CV and then calibrated on the full cohort; deployment artifacts are for inference, not an unbiased performance estimate. No patient-level fusion performance is claimed.
