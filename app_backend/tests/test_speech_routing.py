"""Speech recording modes, saved speech results and where they are shown.

Each mode is tested on its own:
* reading passage: features are extracted and the trained reading model gives the prediction;
* sustained vowel: features are extracted, and the prediction is reported as unavailable (no trained model).

Feature extraction and prediction are separate states everywhere, so features are never shown as a classification.
"""
from __future__ import annotations

import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_backend import create_app
from app_backend.database import Database
from app_backend.errors import APIError
from app_backend.services import pdf_report
from app_backend.services.assessment_summary import build_assessment_summary
from app_backend.services.speech_modes import (
    RECORDING_MODES,
    normalise_recording_mode,
    recording_mode_label,
    speech_result_view,
)
from app_backend.tests.test_backend import make_voiced_wav

PASSAGE = "Today is a pleasant day. I am speaking at my normal pace and volume."


class RecordingModeMappingTests(unittest.TestCase):
    def test_each_mode_maps_to_its_own_label_and_model_file(self):
        self.assertEqual(RECORDING_MODES["reading"]["label"], "Reading passage")
        self.assertEqual(RECORDING_MODES["reading"]["model_file"], "reading.joblib")
        self.assertEqual(RECORDING_MODES["sustained_vowel"]["model_file"], "sustained_vowel.joblib")
        self.assertIn("vowel", recording_mode_label("sustained_vowel").lower())
        self.assertNotEqual(recording_mode_label("sustained_vowel"), recording_mode_label("reading"))

    def test_only_the_two_known_modes_are_accepted_exactly(self):
        self.assertEqual(normalise_recording_mode("reading"), "reading")
        self.assertEqual(normalise_recording_mode(" sustained_vowel "), "sustained_vowel")
        for bad in ("", None, "Reading", "reading passage", "vowel", "sustained"):
            with self.subTest(value=bad), self.assertRaises(APIError) as raised:
                normalise_recording_mode(bad)
            self.assertEqual(raised.exception.code, "invalid_speech_mode")

    def test_unknown_mode_never_falls_back_to_a_label(self):
        self.assertIsNone(recording_mode_label("sustained"))
        self.assertIsNone(speech_result_view({"status": "unavailable", "outcome": {"recording_mode": "other"}})["recording_mode_label"])


class SpeechResultStatesTests(unittest.TestCase):
    def test_unavailable_sustained_vowel_keeps_extraction_and_prediction_apart(self):
        view = speech_result_view({
            "status": "unavailable",
            "outcome": {"feature_extraction": "completed", "feature_count": 45, "feature_schema": "speech_audio_v2",
                        "recording_mode": "sustained_vowel", "prediction_status": "unavailable",
                        "blocker_code": "speech_audio_v2_model_not_trained"},
        })
        self.assertEqual(view["feature_extraction"], "completed")
        self.assertEqual(view["feature_count"], 45)
        self.assertEqual(view["prediction"], "unavailable")
        self.assertIsNone(view["probability"])
        self.assertIn("no prediction was made", view["prediction_reason"])
        self.assertEqual(view["recording_mode_label"], 'Sustained "ah" vowel')

    def test_probability_is_a_prediction_only_when_verified_and_completed(self):
        completed = speech_result_view({"status": "completed", "outcome": {
            "recording_mode": "reading", "feature_extraction": "completed", "feature_count": 45,
            "probability": 0.31, "prediction": 0, "model_verified": True}})
        self.assertEqual(completed["prediction"], "completed")
        self.assertAlmostEqual(completed["probability"], 0.31)
        self.assertEqual(completed["model"], "speech_audio_v2 reading model")
        unverified = speech_result_view({"status": "completed", "outcome": {
            "recording_mode": "reading", "probability": 0.31, "model_verified": False}})
        self.assertEqual(unverified["prediction"], "not_run")
        self.assertIsNone(unverified["probability"])

    def test_pending_and_failed_steps_do_not_claim_features_or_predictions(self):
        pending = speech_result_view(None)
        self.assertEqual((pending["feature_extraction"], pending["prediction"]), ("not_run", "not_run"))
        failed = speech_result_view({"status": "failed", "error": "Audio is silent.", "outcome": {}})
        self.assertEqual(failed["feature_extraction"], "failed")
        self.assertEqual(failed["prediction"], "not_run")
        self.assertIn("Audio is silent.", failed["prediction_reason"])


class SpeechRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pd-speech-routing-")
        self.app = create_app({
            "TESTING": True,
            "DATABASE_PATH": str(Path(self.temp.name) / "test.db"),
            "UPLOAD_TEMP_DIR": self.temp.name,
            "AUDIO_MAX_BYTES": 512 * 1024,
            "SPIRAL_MAX_BYTES": 2 * 1024 * 1024,
            "MAX_CONTENT_LENGTH": 3 * 1024 * 1024,
            "SPIRAL_MEDIA_DIR": str(Path(self.temp.name) / "private-media"),
            "APP_SECRET_KEY": "test_secret_key_12345",
            "SECRET_KEY": "test_secret_key_12345",
        })
        self.client = self.app.test_client()
        self.db = Database(str(Path(self.temp.name) / "test.db"))
        self.app.extensions["db"] = self.db
        from app_backend.services.auth import create_token, hash_password
        self.user = self.db.create_user("speechroute", "speechroute@example.com", hash_password("password123"))
        self.token = create_token(self.user["id"], "test_secret_key_12345")
        self.assessment = self.db.create_assessment(self.user["id"])
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self):
        import shutil
        import time
        try:
            self.db._get_conn().close()
        except Exception:
            pass
        time.sleep(0.1)
        shutil.rmtree(self.temp.name, ignore_errors=True)

    def _submit(self, mode, **extra):
        data = {"mode": mode, "gender_code": "0", **extra,
                "file": (io.BytesIO(make_voiced_wav()), "sample.wav", "audio/wav")}
        return self.client.post(f"/api/assessments/{self.assessment['id']}/speech/predict",
                                data=data, headers=self.headers)

    def _saved_speech(self):
        return self.db.get_assessment(self.assessment["id"], self.user["id"])["steps"]["speech"]

    def test_reading_passage_is_extracted_and_predicted_with_the_reading_model(self):
        response = self._submit("reading", passage_version="1.0", passage_text=PASSAGE)
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual(body["status"], "completed")
        outcome = body["outcome"]
        self.assertEqual(outcome["recording_mode"], "reading")
        self.assertEqual(outcome["feature_extraction"], "completed")
        self.assertEqual(outcome["feature_count"], 45)
        self.assertEqual(outcome["prediction_status"], "completed")
        self.assertTrue(outcome["model_verified"])
        self.assertGreaterEqual(outcome["probability"], 0.0)
        self.assertLessEqual(outcome["probability"], 1.0)
        saved = self._saved_speech()
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["outcome"]["recording_mode"], "reading")
        self.assertEqual(saved["outcome"]["passage_version"], "1.0")

    def test_sustained_vowel_extracts_features_and_reports_prediction_unavailable(self):
        response = self._submit("sustained_vowel")
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual(body["status"], "unavailable")
        outcome = body["outcome"]
        self.assertEqual(outcome["recording_mode"], "sustained_vowel")
        self.assertEqual(outcome["feature_extraction"], "completed")
        self.assertEqual(outcome["feature_count"], 45)
        self.assertEqual(outcome["prediction_status"], "unavailable")
        self.assertNotIn("probability", outcome)
        saved = self._saved_speech()
        self.assertEqual(saved["status"], "unavailable")
        self.assertEqual(saved["outcome"]["recording_mode"], "sustained_vowel")

    def test_second_submission_replaces_the_first_mode_without_stale_fields(self):
        self.assertEqual(self._submit("reading", passage_version="1.0", passage_text=PASSAGE).status_code, 200)
        self.assertEqual(self._submit("sustained_vowel").status_code, 200)
        saved = self._saved_speech()
        self.assertEqual(saved["outcome"]["recording_mode"], "sustained_vowel")
        self.assertIsNone(saved["outcome"].get("passage_version"))
        self.assertNotIn("probability", saved["outcome"])
        self.assertFalse(saved["outcome"]["model_verified"])

    def test_invalid_mode_is_rejected_and_does_not_leave_the_step_processing(self):
        response = self._submit("Reading")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_speech_mode")
        self.assertNotEqual(self._saved_speech()["status"], "processing")

    def test_summary_reports_each_mode_under_its_own_label(self):
        self._submit("reading", passage_version="1.0", passage_text=PASSAGE)
        summary = build_assessment_summary(self.db.get_assessment(self.assessment["id"], self.user["id"]))
        speech = next(item for item in summary["modalities"] if item["modality"] == "speech")
        self.assertEqual(speech["speech_result"]["recording_mode_label"], "Reading passage")
        self.assertEqual(speech["speech_result"]["prediction"], "completed")
        self.assertTrue(speech["prediction_available"])
        self.assertTrue(any("reading passage recording" in item["text"] for item in summary["experimental_observations"]))

    def test_summary_and_pdf_report_sustained_vowel_as_features_only(self):
        self._submit("sustained_vowel")
        assessment = self.db.get_assessment(self.assessment["id"], self.user["id"])
        summary = build_assessment_summary(assessment)
        speech = next(item for item in summary["modalities"] if item["modality"] == "speech")
        self.assertEqual(speech["speech_result"]["recording_mode_label"], 'Sustained "ah" vowel')
        self.assertFalse(speech["prediction_available"])
        self.assertEqual(speech["speech_result"]["feature_extraction"], "completed")
        self.assertNotIn("Reading passage", str(speech["speech_result"]))

        recorded = []
        original = pdf_report.Paragraph

        def recording_paragraph(text, *args, **kwargs):
            recorded.append(str(text))
            return original(text, *args, **kwargs)

        with mock.patch.object(pdf_report, "Paragraph", side_effect=recording_paragraph):
            report = self.client.get(f"/api/assessments/{self.assessment['id']}/report.pdf", headers=self.headers)
        self.assertEqual(report.status_code, 200)
        self.assertTrue(report.data.startswith(b"%PDF"))
        text = "\n".join(recorded)
        self.assertIn("Recording task:</b> Sustained", text)
        self.assertIn("Feature extraction:</b> completed, 45", text)
        self.assertIn("Prediction:</b> unavailable", text)
        self.assertNotIn("Prediction:</b> Parkinson", text)
        self.assertNotIn("Standardized Reading Passage", text)

    def test_pdf_reports_the_reading_prediction_with_its_probability(self):
        self._submit("reading", passage_version="1.0", passage_text=PASSAGE)
        recorded = []
        original = pdf_report.Paragraph

        def recording_paragraph(text, *args, **kwargs):
            recorded.append(str(text))
            return original(text, *args, **kwargs)

        with mock.patch.object(pdf_report, "Paragraph", side_effect=recording_paragraph):
            report = self.client.get(f"/api/assessments/{self.assessment['id']}/report.pdf", headers=self.headers)
        self.assertEqual(report.status_code, 200)
        text = "\n".join(recorded)
        self.assertIn("Recording task:</b> Reading passage", text)
        self.assertIn("Feature extraction:</b> completed, 45", text)
        self.assertIn("Prediction:</b>", text)
        self.assertRegex(text, r"probability <b>\d+\.\d{2}%</b>")
        self.assertIn("from the speech_audio_v2 reading model", text)


class SpeechResultRoutingTests(unittest.TestCase):
    """The Results card title follows the saved recording mode (checked in the frontend tests too)."""

    def test_speech_summary_title_comes_from_the_saved_mode_not_a_fixed_label(self):
        for mode, label in (("reading", "Reading passage"), ("sustained_vowel", 'Sustained "ah" vowel')):
            with self.subTest(mode=mode):
                view = speech_result_view({"status": "unavailable", "outcome": {
                    "recording_mode": mode, "feature_extraction": "completed", "feature_count": 45}})
                self.assertEqual(view["recording_mode_label"], label)


if __name__ == "__main__":
    unittest.main()


class SpeechFeatureDisplayCoverageTests(unittest.TestCase):
    """The Speech page shows every extracted feature, and only the names the backend returns."""

    def test_every_extracted_feature_is_displayed_in_a_group(self):
        from app_backend.services.acoustic_features import SPEECH_V2_FEATURES

        source = (Path(__file__).resolve().parents[2] / "frontend" / "src" / "components" / "SpeechFeatureGroups.tsx").read_text(encoding="utf-8")
        literal_keys = set(re.findall(r"key: '(audio_v2_[a-z0-9_]+)'", source))
        generated_keys = {f"audio_v2_mfcc{i}_mean" for i in range(13)} | {f"audio_v2_delta{i}_mean" for i in range(13)}
        displayed = literal_keys | generated_keys
        self.assertEqual(sorted(set(SPEECH_V2_FEATURES) - displayed), [], "extracted features missing from the page")
        self.assertEqual(sorted(displayed - set(SPEECH_V2_FEATURES)), [], "page shows names the backend does not return")
        self.assertEqual(len(SPEECH_V2_FEATURES), 45)
