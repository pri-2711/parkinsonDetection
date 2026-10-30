from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app_backend import create_app
from app_backend.database import Database
from app_backend.repositories.assessment_repository import AssessmentRepository
from app_backend.services.acoustic_features import SPEECH_V2_FEATURES, SPEECH_V2_MODEL_FEATURES
from app_backend.services.assessment_summary import build_assessment_summary
from app_backend.services.auth import create_token, hash_password
from app_backend.services.pdf_report import generate_assessment_pdf
from app_backend.services.research_insights import build_research_insights
from app_backend.services.speech_v2_model import SpeechV2Model
from app_backend.tests.test_backend import make_voiced_wav

ROOT = Path(__file__).resolve().parents[2]
SECRET = "test_secret_key_12345"


def _labelled_wav(label: int, seed: int) -> bytes:
    """Synthetic voice-like tone; label 1 gets more cycle-to-cycle perturbation. Not real speech."""
    import soundfile as sf
    rng = np.random.default_rng(seed)
    times = np.arange(int(5 * 16000), dtype=float) / 16000
    knot_times = np.linspace(0, 5, 51)
    perturbation = (0.02 + 0.03 * label) * rng.standard_normal(knot_times.size)
    f0 = 170 * (1 + np.interp(times, knot_times, perturbation))
    phase = 2 * np.pi * np.cumsum(f0) / 16000
    signal = 0.2 * np.sin(phase) + 0.003 * rng.normal(size=times.size)
    stream = io.BytesIO()
    sf.write(stream, signal, 16000, format="WAV", subtype="PCM_16")
    return stream.getvalue()


class LegacyCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pd-speech-compat-")
        cls.app = create_app({
            "TESTING": True,
            "UPLOAD_TEMP_DIR": cls.temp.name,
            "DATABASE_PATH": str(Path(cls.temp.name) / "compat.db"),
            "APP_SECRET_KEY": SECRET,
            "SECRET_KEY": SECRET,
        })

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_legacy_svm_rejects_speech_v2_columns(self):
        adapter = self.app.extensions["model_registry"].models["speech"]
        v2_frame = pd.DataFrame([np.zeros(len(SPEECH_V2_FEATURES))], columns=SPEECH_V2_FEATURES)
        with self.assertRaises(ValueError):
            adapter.predict(v2_frame)
        self.assertEqual(adapter.features[0], "gender")
        self.assertTrue(set(SPEECH_V2_FEATURES).isdisjoint(adapter.features))

    def test_v2_model_reports_not_trained_when_no_artifact_exists(self):
        model = SpeechV2Model(Path(self.temp.name) / "empty", "sustained_vowel")
        self.assertFalse(model.ready)
        self.assertEqual(model.error_code, "speech_audio_v2_model_not_trained")

    def test_v2_model_rejects_wrong_schema_bundle(self):
        folder = Path(self.temp.name) / "bad"
        folder.mkdir()
        joblib.dump({"schema": "legacy", "mode": "sustained_vowel", "features": ["gender"], "model": None},
                    folder / "sustained_vowel.joblib")
        model = SpeechV2Model(folder, "sustained_vowel")
        self.assertFalse(model.ready)
        self.assertEqual(model.error_code, "speech_audio_v2_model_invalid")

    def test_v2_model_rejects_reordered_feature_frame(self):
        from sklearn.dummy import DummyClassifier
        folder = Path(self.temp.name) / "order"
        folder.mkdir()
        classifier = DummyClassifier(strategy="prior").fit(
            np.zeros((4, len(SPEECH_V2_MODEL_FEATURES))), [0, 0, 1, 1])
        joblib.dump({"schema": "speech_audio_v2", "mode": "reading", "features": SPEECH_V2_MODEL_FEATURES, "model": classifier},
                    folder / "reading.joblib")
        model = SpeechV2Model(folder, "reading")
        self.assertTrue(model.ready)
        reordered = pd.DataFrame([np.zeros(len(SPEECH_V2_FEATURES))], columns=list(reversed(SPEECH_V2_FEATURES)))
        with self.assertRaises(ValueError):
            model.predict(reordered)

    def test_extractor_output_is_ordered_45_column_v2_schema(self):
        from app_backend.services.acoustic_features import AcousticFeatureExtractor
        frame = AcousticFeatureExtractor().extract(make_voiced_wav(), {}, mode="sustained_vowel", gender_code=1)
        self.assertEqual(list(frame.columns), SPEECH_V2_FEATURES)
        self.assertEqual(len(SPEECH_V2_FEATURES), 45)
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())


class TrainingLeakageTests(unittest.TestCase):
    def test_outer_folds_are_subject_disjoint_and_contain_both_classes(self):
        from experiments.speech_audio_v2.train import outer_folds
        groups = np.array([f"S{i:02d}" for i in range(12) for _ in range(2)])
        y = np.array([i % 2 for i in range(12) for _ in range(2)])
        folds = outer_folds(y, groups)
        self.assertEqual(len(folds), 5)
        for train, test in folds:
            self.assertEqual(set(groups[train]) & set(groups[test]), set())
            self.assertEqual(set(y[train]), {0, 1})
            self.assertEqual(set(y[test]), {0, 1})

    def test_outer_folds_refuse_fewer_than_eight_subjects(self):
        from experiments.speech_audio_v2.train import outer_folds
        with self.assertRaisesRegex(ValueError, "at least 8"):
            outer_folds(np.array([0, 0, 1, 1]), np.array(["a", "a", "b", "b"]))

class TrainedSpeechPathTests(unittest.TestCase):
    """Runs the real train -> load -> predict -> persist -> finalize chain on synthetic audio.

    The model written here is a plumbing check in a temporary folder. It is not a
    clinical model and is never written to the repository's artifact directory.
    """

    @classmethod
    def setUpClass(cls):
        from experiments.speech_audio_v2.train import train
        cls.temp = tempfile.TemporaryDirectory(prefix="pd-speech-e2e-")
        base = Path(cls.temp.name)
        audio_dir = base / "audio"
        audio_dir.mkdir()
        rows = []
        for subject in range(12):
            label = subject % 2
            for take in range(2):
                name = f"s{subject:02d}_{take}.wav"
                (audio_dir / name).write_bytes(_labelled_wav(label, seed=subject * 10 + take))
                rows.append({"audio_path": f"audio/{name}", "subject_id": f"S{subject:02d}",
                             "mode": "reading", "gender_code": subject % 2, "status": label})
        manifest = base / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        cls.artifacts = base / "artifacts"
        cls.metrics = train(manifest, cls.artifacts)
        cls.app = create_app({
            "TESTING": True,
            "UPLOAD_TEMP_DIR": cls.temp.name,
            "DATABASE_PATH": str(base / "e2e.db"),
            "AUDIO_MAX_BYTES": 8 * 1024 * 1024,
            "MAX_CONTENT_LENGTH": 9 * 1024 * 1024,
            "SPEECH_AUDIO_V2_ARTIFACT_DIR": str(cls.artifacts),
            "APP_SECRET_KEY": SECRET,
            "SECRET_KEY": SECRET,
        })
        cls.db = Database(str(base / "e2e.db"))
        cls.app.extensions["db"] = cls.db
        cls.client = cls.app.test_client()
        cls.user = cls.db.create_user("e2euser", "e2e@example.com", hash_password("password123"))
        cls.token = create_token(cls.user["id"], SECRET)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.db._get_conn().close()
        except Exception:
            pass
        cls.temp.cleanup()

    def test_training_wrote_only_to_requested_folder_and_reports_holdout(self):
        self.assertTrue((self.artifacts / "reading.joblib").is_file())
        self.assertFalse((self.artifacts / "sustained_vowel.joblib").exists())
        self.assertNotEqual(self.artifacts.resolve(), (ROOT / "experiments" / "speech_audio_v2" / "artifacts").resolve())
        result = self.metrics["reading"]
        self.assertEqual(result["n_subjects"], 12)
        self.assertTrue(result["subject_disjoint_folds"])
        self.assertEqual(result["outer_folds"], 5)
        self.assertIn(result["deployed_family"], ("svm", "random_forest"))
        self.assertIn("subject", result["validation"])

    def test_trained_mode_predicts_and_persists_and_untrained_mode_stays_unavailable(self):
        assessment = self.db.create_assessment(self.user["id"])
        response = self.client.post(
            f"/api/assessments/{assessment['id']}/speech/predict",
            data={"mode": "reading", "gender_code": "1",
                  "file": (io.BytesIO(_labelled_wav(1, seed=999)), "reading.wav", "audio/wav")},
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual(body["status"], "completed")
        self.assertTrue(0.0 <= body["outcome"]["probability"] <= 1.0)
        stored = self.db.get_assessment(assessment["id"], self.user["id"])["steps"]["speech"]
        self.assertEqual(stored["status"], "completed")
        self.assertTrue(stored["outcome"]["model_verified"])
        self.assertEqual(stored["outcome"]["feature_schema"], "speech_audio_v2")
        self.assertEqual(len(stored["outcome"]["feature_values"]), 45)
        self.assertEqual(stored["outcome"]["recording_mode"], "reading")
        self.assertNotIn("audio_bytes", str(stored))

        # sustained_vowel has no trained model in this test bundle, so it must stay unavailable.
        vowel = self.client.post(
            f"/api/assessments/{assessment['id']}/speech/predict",
            data={"mode": "sustained_vowel", "gender_code": "1",
                  "file": (io.BytesIO(_labelled_wav(1, seed=1000)), "vowel.wav", "audio/wav")},
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(vowel.status_code, 200)
        self.assertEqual(vowel.get_json()["status"], "unavailable")
        after = self.db.get_assessment(assessment["id"], self.user["id"])["steps"]["speech"]
        self.assertEqual(after["status"], "unavailable")
        self.assertEqual(after["outcome"]["blocker_code"], "speech_audio_v2_model_not_trained")

    def test_finalized_single_modality_result_shows_probability_and_summary_matches(self):
        assessment = self.db.create_assessment(self.user["id"])
        self.client.post(
            f"/api/assessments/{assessment['id']}/speech/predict",
            data={"mode": "reading", "gender_code": "0",
                  "file": (io.BytesIO(_labelled_wav(0, seed=321)), "vowel.wav", "audio/wav")},
            headers={"Authorization": f"Bearer {self.token}"},
        )
        step = self.db.get_assessment(assessment["id"], self.user["id"])["steps"]["speech"]
        complete = self.client.post(f"/api/assessments/{assessment['id']}/complete",
                                    headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(complete.status_code, 200, complete.get_json())
        stored = self.db.get_assessment(assessment["id"], self.user["id"])
        self.assertEqual(stored["status"], "completed")
        self.assertEqual(stored["results"]["fusion"]["available_modalities"], ["speech"])
        summary = build_assessment_summary(stored)
        self.assertEqual(summary["fusion"]["state"], "single_modality")
        self.assertTrue(summary["fusion"]["fused_probability_displayed"])
        self.assertAlmostEqual(summary["fusion"]["single_modality_probability"], step["outcome"]["probability"])

        summary_response = self.client.get(f"/api/assessments/{assessment['id']}/summary",
                                           headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(summary_response.status_code, 200)
        self.assertEqual(summary_response.get_json()["fusion"]["state"], "single_modality")

        other = self.db.create_user("otheruser", "other@example.com", hash_password("password123"))
        other_token = create_token(other["id"], SECRET)
        denied = self.client.get(f"/api/assessments/{assessment['id']}/summary",
                                 headers={"Authorization": f"Bearer {other_token}"})
        self.assertEqual(denied.status_code, 404)


class AssessmentSummaryRuleTests(unittest.TestCase):
    def _assessment(self, status="in_progress", steps=None, results=None):
        base_steps = {
            "speech": {"status": "pending", "outcome": None, "error": None},
            "spiral": {"status": "pending", "outcome": None, "error": None},
            "tremor": {"status": "pending", "outcome": None, "error": None},
        }
        base_steps.update(steps or {})
        return {"id": 7, "status": status, "steps": base_steps, "results": results}

    def test_unavailable_speech_is_observation_not_prediction(self):
        summary = build_assessment_summary(self._assessment(steps={
            "speech": {"status": "unavailable", "error": "no model", "outcome": {
                "feature_extraction": "completed", "feature_count": 45, "feature_schema": "speech_audio_v2",
                "recording_mode": "sustained_vowel", "prediction_status": "unavailable",
                "blocker_code": "speech_audio_v2_model_not_trained"}},
        }))
        speech = next(item for item in summary["modalities"] if item["modality"] == "speech")
        self.assertFalse(speech["prediction_available"])
        self.assertTrue(speech["processed"])
        self.assertIn("45 speech_audio_v2 features", speech["observations"][0])
        self.assertEqual(summary["predictions_available"], [])
        self.assertEqual(summary["fusion"]["state"], "not_finalized")

    def test_multi_modality_fusion_hides_combined_probability(self):
        steps = {
            "tremor": {"status": "completed", "outcome": {"probability": 0.8, "model_verified": True}},
            "spiral": {"status": "completed", "outcome": {"probability": 0.4, "model_verified": True}},
        }
        results = {"fusion": {"probability": 0.6, "available_modalities": ["tremor", "handwriting"],
                              "effective_weights": {"tremor": 0.5, "handwriting": 0.5}}}
        summary = build_assessment_summary(self._assessment(status="completed", steps=steps, results=results))
        self.assertEqual(summary["fusion"]["state"], "multi_modality_aggregate")
        self.assertFalse(summary["fusion"]["fused_probability_displayed"])
        self.assertIsNone(summary["fusion"]["single_modality_probability"])
        self.assertEqual(len(summary["fusion"]["contributing_modalities"]), 2)
        self.assertEqual(len(summary["predictions_available"]), 2)

    def test_unverified_probability_is_never_counted_as_prediction(self):
        steps = {"tremor": {"status": "completed", "outcome": {"probability": 0.9}}}
        summary = build_assessment_summary(self._assessment(status="completed", steps=steps,
                                                            results={"fusion": None}))
        tremor = next(item for item in summary["modalities"] if item["modality"] == "tremor")
        self.assertFalse(tremor["prediction_available"])
        self.assertFalse(tremor["contributes_to_fusion"])


class ResearchAndReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pd-report-tests-")
        cls.app = create_app({
            "TESTING": True,
            "UPLOAD_TEMP_DIR": cls.temp.name,
            "DATABASE_PATH": str(Path(cls.temp.name) / "report.db"),
            "APP_SECRET_KEY": SECRET,
            "SECRET_KEY": SECRET,
        })
        cls.db = Database(str(Path(cls.temp.name) / "report.db"))
        cls.app.extensions["db"] = cls.db
        cls.client = cls.app.test_client()
        cls.user = cls.db.create_user("reportuser", "report@example.com", hash_password("password123"))
        cls.token = create_token(cls.user["id"], SECRET)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.db._get_conn().close()
        except Exception:
            pass
        cls.temp.cleanup()

    def test_research_insights_match_saved_tuning_report(self):
        results = ROOT / "experiments" / "regularized_svm_rf" / "results"
        insights = build_research_insights(results, results / "artifacts",
                                           ROOT / "experiments" / "speech_audio_v2" / "artifacts")
        speech = next(item for item in insights["modalities"] if item["modality"] == "speech")
        # TUNING_REPORT.md: speech tuned SVM, folds 2-5, mean ± sample SD = 0.7604 ± 0.0269 (accuracy)
        accuracy = speech["fold_metrics"]["tuned_svm"]["accuracy"]
        self.assertAlmostEqual(accuracy["mean"], 0.7604, places=4)
        self.assertAlmostEqual(accuracy["sd"], 0.0269, places=4)
        self.assertAlmostEqual(speech["pooled_metrics"]["tuned_svm"]["roc_auc"], 0.8238, places=4)
        self.assertEqual(speech["rows"], 240)
        self.assertEqual(speech["participant_groups"], 80)
        speech_v2 = insights["speech_audio_v2"]
        self.assertEqual(speech_v2["status"], "experimental_prediction")
        self.assertTrue(speech_v2["modes"]["reading"]["model_trained"])
        self.assertFalse(speech_v2["modes"]["sustained_vowel"]["model_trained"])
        self.assertEqual(speech_v2["modes"]["reading"]["subjects"], 37)
        self.assertEqual(speech_v2["feature_count"], 44)
        self.assertTrue(speech_v2["limitations"])
        self.assertIsNone(insights["fusion"]["patient_level_performance"])
        self.assertAlmostEqual(sum(insights["fusion"]["weights"].values()), 1.0, places=6)

    def test_research_endpoint_requires_auth_and_returns_insights(self):
        self.assertEqual(self.client.get("/api/research/insights").status_code, 401)
        response = self.client.get("/api/research/insights", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual({m["modality"] for m in body["modalities"]}, {"speech", "handwriting", "tremor"})

    def test_pdf_report_is_downloadable_with_summary_and_access_control(self):
        assessment = self.db.create_assessment(self.user["id"])
        AssessmentRepository(self.db).update_assessment_step(assessment["id"], self.user["id"], "tremor", {
            "status": "completed", "outcome": {"probability": 0.7, "prediction": 1, "model_verified": True},
            "error": None,
        })
        response = self.client.get(f"/api/assessments/{assessment['id']}/report.pdf",
                                   headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertTrue(response.data.startswith(b"%PDF"))
        self.assertGreater(len(response.data), 4000)

        denied = self.client.get(f"/api/assessments/{assessment['id']}/report.pdf")
        self.assertEqual(denied.status_code, 401)

    def test_pdf_generates_for_empty_and_failed_assessments(self):
        empty = {"id": 1, "status": "in_progress", "steps": {}, "results": None}
        self.assertTrue(generate_assessment_pdf(empty, None, None).startswith(b"%PDF"))
        failed = {"id": 2, "status": "in_progress", "steps": {
            "speech": {"status": "failed", "error": "Audio processing failed.", "outcome": {}},
        }, "results": None}
        self.assertTrue(generate_assessment_pdf(failed, None, None).startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
