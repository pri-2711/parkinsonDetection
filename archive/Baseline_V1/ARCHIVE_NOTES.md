# Baseline V1 archive

This archive is a snapshot of the working pipeline and data as they existed at freeze time. It does not include `.git`; the working tree had uncommitted changes, so the archived files—not the commit alone—define Baseline V1.

- Source commit: `9e706cd4aaffcdbcf96801d74beff6746d7f0616`
- Source branch: `model_v1`
- Captured working changes: README and cleaning scripts, handwriting feature extractor/table, plus untracked current `src/`, `scripts/`, `tests/`, `models/`, and `reports/` trees.
- `report.md` was absent at freeze time and is not reconstructed here.
- Raw input data, cleaned data, feature table, all files under `models/artifacts/`, reports, code, tests, and requirements are copied into this archive.
- `ARCHIVE_MANIFEST.json` records SHA-256 hashes for archived files and package/runtime versions.
- Run `python verify_archive.py` from any directory to validate archived hashes, load the saved model bundles, run sample inference from archived feature rows, and recompute the saved OOF metrics. This verification does not refit models.

Baseline configuration uses random seed 42, five outer subject-grouped folds for classification and regression, three group-aware calibration folds, and sigmoid calibration. See `models/artifacts/run_config.json` for the full pipeline configuration and `models/artifacts/fusion_config.json` for the late-fusion weights and method.
