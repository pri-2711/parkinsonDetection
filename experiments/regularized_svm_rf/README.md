# Regularized SVM and Random Forest experiments

This is an isolated experiment workspace created after Baseline V1 was archived and verified. Baseline V1 is preserved at `../../archive/Baseline_V1`; experiment code and outputs belong here and must not overwrite that archive or the current baseline artifacts.

## Planned experiment

- Reuse the three modality datasets and subject-group definitions from the archived Baseline V1 configuration.
- Retain RBF SVM and Random Forest as the only model families. Explore a modest, predeclared grid for SVM regularization (`C`/`gamma`) and Random Forest complexity (`max_depth`, `min_samples_leaf`, and `max_features`).
- Use nested, stratified group-aware validation, fit imputation/scaling/calibration inside training folds, and select hyperparameters on inner-fold ROC-AUC. Report F1 and recall at the fixed 0.5 threshold.
- Keep telemonitoring separate from classification and do not attempt cross-cohort patient matching or multimodal fusion evaluation.
- Write all experiment predictions, models, metrics, and configuration below this directory.

`experiment_config.json` records the predeclared search space and evaluation plan. Run from the repository root:

```powershell
python experiments/regularized_svm_rf/run_experiments.py
```

This runs separate nested grouped searches for SVM and Random Forest on speech, handwriting, and tremor. It reads the immutable Baseline V1 snapshot and writes predictions, metrics, and deployable tuned model bundles only below `results/`. Five-fold subject-grouped splits and inner four-fold parameter selection use seed 42; sigmoid calibration uses three group-aware folds. For a separate fold reserved for fusion-weight validation, report metrics use the other four outer folds for baseline-versus-tuned comparisons.

Then run:

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/regularized_svm_rf -p 'test_*.py' -v
python archive/Baseline_V1/verify_archive.py
```

See `results/TUNING_REPORT.md` for results, selected models, limitations, and external validation status. No suitable independent local cohort is present; patient-level fusion performance is not validated.
