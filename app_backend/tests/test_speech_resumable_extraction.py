from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app_backend.services.acoustic_features import SPEECH_V2_FEATURES, SPEECH_V2_MODEL_FEATURES
from experiments.speech_audio_v2.train import CACHE_HEADER, extract_features


class StubExtractor:
    """Deterministic stand-in for AcousticFeatureExtractor. `interrupt_after` simulates a killed process."""

    def __init__(self, interrupt_after: int | None = None, fail_names: set[str] | None = None):
        self.calls = 0
        self.interrupt_after = interrupt_after
        self.fail_names = fail_names or set()

    def extract(self, audio_bytes, metadata, *, mode, gender_code):
        self.calls += 1
        if self.interrupt_after is not None and self.calls > self.interrupt_after:
            raise KeyboardInterrupt  # BaseException: not swallowed by the per-recording handler
        name = audio_bytes.decode()
        if name in self.fail_names:
            from app_backend.errors import APIError
            raise APIError(422, "silent_audio", "silent")
        value = float(len(name)) + gender_code
        row = {feature: value + index for index, feature in enumerate(SPEECH_V2_FEATURES)}
        return pd.DataFrame([row], columns=SPEECH_V2_FEATURES)


class ResumableExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="pd-resume-"))
        self.rows = []
        for subject in range(4):
            for take in range(2):
                name = f"s{subject}_{take}.wav"
                (self.temp / name).write_bytes(name.encode())
                self.rows.append({"audio_path": str(self.temp / name), "subject_id": f"S{subject}",
                                  "mode": "reading", "gender_code": 0, "status": subject % 2})
        self.manifest = pd.DataFrame(self.rows)
        self.cache = self.temp / "cache" / "features_reading.csv"

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def _run(self, extractor):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = extract_features(self.manifest, extractor, "reading", self.cache)
        return result, out.getvalue()

    def test_interrupted_run_keeps_completed_recordings_and_resume_skips_them(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(StubExtractor(interrupt_after=3))
        saved = pd.read_csv(self.cache)
        self.assertEqual(len(saved), 3)
        self.assertEqual(list(pd.read_csv(self.cache, nrows=0).columns), CACHE_HEADER)

        second = StubExtractor()
        (features, kept, skipped), log = self._run(second)
        self.assertEqual(second.calls, 5)  # 8 recordings minus 3 cached
        self.assertEqual(len(features), 8)
        self.assertEqual(skipped, [])
        self.assertEqual(len(pd.read_csv(self.cache)), 8)
        self.assertIn("3 from cache", log)

    def test_resume_output_equals_uninterrupted_output(self):
        (fresh, _, _), _ = self._run(StubExtractor())
        self.cache.unlink()
        with self.assertRaises(KeyboardInterrupt):
            self._run(StubExtractor(interrupt_after=5))
        (resumed, _, _), _ = self._run(StubExtractor())
        pd.testing.assert_frame_equal(fresh, resumed)

    def test_subject_and_label_stay_with_each_recording(self):
        (features, kept, _), _ = self._run(StubExtractor())
        expected = self.manifest.set_index("audio_path")
        for record in kept.itertuples(index=False):
            self.assertEqual(record.subject_id, expected.loc[record.audio_path, "subject_id"])
            self.assertEqual(int(record.status), int(expected.loc[record.audio_path, "status"]))
        self.assertEqual(list(features.columns), SPEECH_V2_MODEL_FEATURES)

    def test_partial_last_line_is_discarded_and_re_extracted(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(StubExtractor(interrupt_after=2))
        with self.cache.open("a", encoding="utf-8") as handle:
            handle.write("S9_0.wav,S9,reading,0,1,12")  # killed mid-line, no newline
        (features, kept, _), _ = self._run(StubExtractor())
        self.assertEqual(len(features), 8)
        self.assertFalse(kept["audio_path"].astype(str).str.contains("S9").any())
        self.assertEqual(len(pd.read_csv(self.cache)), 8)

    def test_cache_with_wrong_feature_schema_is_rejected(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"audio_path": [self.rows[0]["audio_path"]], "subject_id": ["S0"]}).to_csv(self.cache, index=False)
        with self.assertRaisesRegex(ValueError, "does not match the expected"):
            self._run(StubExtractor())

    def test_duplicate_recordings_in_cache_are_rejected(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(StubExtractor(interrupt_after=2))
        lines = self.cache.read_text(encoding="utf-8").splitlines()
        self.cache.write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._run(StubExtractor())

    def test_cache_label_mismatch_with_manifest_is_rejected(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(StubExtractor(interrupt_after=2))
        table = pd.read_csv(self.cache)
        table.loc[0, "status"] = 1 - int(table.loc[0, "status"])
        table.to_csv(self.cache, index=False)
        with self.assertRaisesRegex(ValueError, "differs from the manifest"):
            self._run(StubExtractor())

    def test_failed_recordings_are_logged_not_cached_and_progress_is_printed(self):
        failing = self.rows[1]["audio_path"]
        extractor = StubExtractor(fail_names={Path(failing).name.encode().decode()})
        (features, kept, skipped), log = self._run(extractor)
        self.assertEqual(len(features), 7)
        self.assertEqual(skipped[0]["code"], "silent_audio")
        self.assertNotIn(failing, set(pd.read_csv(self.cache)["audio_path"].astype(str)))
        self.assertIn("[1/8]", log)
        self.assertIn("[8/8]", log)
        self.assertIn("skipped (silent_audio)", log)


if __name__ == "__main__":
    unittest.main()
