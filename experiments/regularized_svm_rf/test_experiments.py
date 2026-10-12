import unittest
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.regularized_svm_rf.run_experiments import (
    _model_and_grid,
    group_splits,
    load_modality,
    reported_test_indices,
)
from src.fusion import fuse_probabilities


class NestedTuningProtocolTests(unittest.TestCase):
    def test_group_splits_keep_participants_separate(self):
        x = pd.DataFrame({"x": np.arange(40, dtype=float)})
        y = np.tile([0, 1], 20)
        groups = np.repeat(np.arange(20), 2)
        splits = group_splits(x, y, groups, 5)
        for train, valid in splits:
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            self.assertEqual(set(y[valid]), {0, 1})

    def test_fusion_validation_fold_is_excluded_from_model_metrics(self):
        assignments = np.repeat(np.arange(1, 6), 2)
        reported = reported_test_indices(assignments, excluded_folds=(1,))
        self.assertTrue(np.all(assignments[reported] != 1))
        self.assertEqual(len(reported), 8)

    def test_predeclared_search_grid_sizes(self):
        from experiments.regularized_svm_rf.run_experiments import _read_config

        config = _read_config()
        svm, svm_grid = _model_and_grid("svm", config)
        rf, rf_grid = _model_and_grid("random_forest", config)
        self.assertEqual(svm.named_steps["imputer"].strategy, "median")
        self.assertEqual(rf.named_steps["imputer"].strategy, "median")
        self.assertEqual(np.prod([len(v) for v in svm_grid.values()]), 18)
        self.assertEqual(np.prod([len(v) for v in rf_grid.values()]), 24)

    def test_saved_models_infer_for_each_modality_and_fuse_available_combinations(self):
        root = Path(__file__).resolve().parent
        result_root = root / "results"
        weights = json.loads((result_root / "fusion_validation_weights.json").read_text(encoding="utf-8"))["normalized_weights"]
        probabilities = {}
        for modality in ("speech", "handwriting", "tremor"):
            x, _, _, features = load_modality(modality)
            bundle = joblib.load(result_root / "artifacts" / f"{modality}_svm_tuned.joblib")
            p = bundle["model"].predict_proba(x.iloc[:3].loc[:, features])[:, 1]
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p >= 0) & (p <= 1)).all())
            probabilities[modality] = p

        modalities = tuple(probabilities)
        for size in range(1, len(modalities) + 1):
            for selected in combinations(modalities, size):
                args = {f"{name}_probability": probabilities[name] for name in selected}
                result = fuse_probabilities(**args, weights=weights)
                self.assertEqual(set(result["available_modalities"]), set(selected))
                self.assertTrue(np.isfinite(result["probability"]).all())
                self.assertTrue(((result["probability"] >= 0) & (result["probability"] <= 1)).all())
                self.assertAlmostEqual(sum(result["effective_weights"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
