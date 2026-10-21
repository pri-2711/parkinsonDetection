"""Versioned raw-audio features. This schema is intentionally separate from legacy SVM."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import butter, hilbert, sosfiltfilt, stft

from app_backend.errors import APIError


# v2 features are named separately because the historical 45-column feature
# recipe is incomplete and must never receive these estimates.
SPEECH_V2_FEATURES = [
    "audio_v2_gender_code",
    "audio_v2_jitter_local", "audio_v2_jitter_abs_s", "audio_v2_jitter_rap", "audio_v2_jitter_ppq5",
    "audio_v2_shimmer_local", "audio_v2_shimmer_db", "audio_v2_shimmer_apq3", "audio_v2_shimmer_apq5", "audio_v2_shimmer_apq11",
    "audio_v2_hnr_500_db", "audio_v2_hnr_1500_db", "audio_v2_hnr_2500_db", "audio_v2_hnr_3500_db", "audio_v2_hnr_3800_db",
    "audio_v2_rpde_return_entropy", "audio_v2_dfa_alpha", "audio_v2_pitch_period_entropy", "audio_v2_gne_max_envelope_corr",
    *[f"audio_v2_mfcc{i}_mean" for i in range(13)],
    *[f"audio_v2_delta{i}_mean" for i in range(13)],
]


@dataclass(frozen=True)
class AudioInput:
    samples: np.ndarray
    sample_rate: int


class AcousticFeatureExtractor:
    """Deterministic v2 extractor for sustained vowels or reading recordings.

    Praat measures use explicit 75-500 Hz pitch bounds and its default CC
    periodicity implementation. Remaining custom features are defined in
    app_backend/FEATURE_COMPATIBILITY.md. They are not legacy-schema proxies.
    """

    schema = "speech_audio_v2"
    features = SPEECH_V2_FEATURES

    @property
    def available(self) -> bool:
        try:
            import soundfile as sf
            import parselmouth  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def read_wav(audio_bytes: bytes) -> AudioInput:
        try:
            import soundfile as sf
            samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float64", always_2d=True)
        except Exception:
            raise APIError(400, "invalid_audio", "The WAV could not be decoded as PCM audio.") from None
        if samples.size == 0 or sample_rate < 8000 or sample_rate > 48000:
            raise APIError(422, "invalid_audio_signal", "Audio must contain samples at 8–48 kHz.")
        mono = np.mean(samples, axis=1)
        if not np.isfinite(mono).all() or np.max(np.abs(mono)) < 1e-5:
            raise APIError(422, "silent_audio", "The recording is silent or has no usable signal.")
        return AudioInput(mono.astype(np.float64), int(sample_rate))

    @staticmethod
    def _dfa_alpha(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size < 64:
            raise ValueError("DFA needs at least 64 voiced pitch periods")
        profile = np.cumsum(values - values.mean())
        scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(values.size // 4), 12)).astype(int))
        sizes, fluctuations = [], []
        for size in scales:
            segments = values.size // size
            if segments < 2:
                continue
            residuals = []
            for segment in range(segments):
                y = profile[segment * size:(segment + 1) * size]
                x = np.arange(size, dtype=float)
                trend = np.polyval(np.polyfit(x, y, 1), x)
                residuals.append(np.mean((y - trend) ** 2))
            sizes.append(size)
            fluctuations.append(np.sqrt(np.mean(residuals)))
        if len(sizes) < 3 or min(fluctuations) <= 0:
            raise ValueError("DFA could not estimate a stable scaling slope")
        return float(np.polyfit(np.log(sizes), np.log(fluctuations), 1)[0])

    @staticmethod
    def _rpde_return_entropy(values: np.ndarray) -> float:
        """Normalized first-return-time entropy for 3D delay vectors (lag=1)."""
        x = np.asarray(values, dtype=float)
        if x.size < 32:
            raise ValueError("RPDE requires at least 32 voiced pitch periods")
        x = (x - np.mean(x)) / (np.std(x) + 1e-12)
        embedded = np.column_stack((x[:-2], x[1:-1], x[2:]))
        radius = 0.1 * np.sqrt(3.0)
        distances = np.linalg.norm(embedded[:, None, :] - embedded[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        returns = []
        for i, row in enumerate(distances):
            candidates = np.flatnonzero(row < radius)
            candidates = candidates[np.abs(candidates - i) > 2]
            if candidates.size:
                returns.append(int(np.min(np.abs(candidates - i))))
        if len(returns) < 8:
            raise ValueError("RPDE could not find enough recurrence returns")
        counts = np.bincount(returns)
        p = counts[counts > 0] / len(returns)
        return float(-np.sum(p * np.log(p)) / np.log(max(2, len(p))))

    @staticmethod
    def _pitch_period_entropy(values: np.ndarray) -> float:
        """Normalized Shannon entropy of 20-bin voiced F0-period histogram."""
        periods = np.asarray(values, dtype=float)
        if periods.size < 32 or not np.isfinite(periods).all():
            raise ValueError("Pitch-period entropy needs at least 32 finite voiced periods")
        counts, _ = np.histogram(periods, bins=20, range=(float(periods.min()), float(periods.max()) + 1e-12))
        probabilities = counts[counts > 0] / periods.size
        return float(-np.sum(probabilities * np.log(probabilities)) / np.log(20))

    @staticmethod
    def _mfcc_and_delta(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """40-band HTK-mel MFCCs: 25 ms Hamming frames, 10 ms hop, DCT-II."""
        frame_length = max(64, int(round(0.025 * sample_rate)))
        hop = max(1, int(round(0.010 * sample_rate)))
        n_fft = 1 << int(np.ceil(np.log2(frame_length)))
        _, _, spectrum = stft(samples, fs=sample_rate, window="hamming", nperseg=frame_length,
                              noverlap=frame_length - hop, nfft=n_fft, boundary="zeros", padded=True)
        power = np.abs(spectrum) ** 2
        mel_max = 2595.0 * np.log10(1.0 + (sample_rate / 2) / 700.0)
        mel_points = np.linspace(0.0, mel_max, 42)
        hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
        bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        bank = np.zeros((40, n_fft // 2 + 1), dtype=float)
        for m in range(1, 41):
            left, center, right = bins[m - 1:m + 2]
            if center > left:
                bank[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                bank[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
        log_mel = np.log(np.maximum(bank @ power, 1e-12))
        mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:13]
        width = 9
        if mfcc.shape[1] < width:
            raise ValueError("At least 9 acoustic frames are required for delta features")
        offsets = np.arange(-(width // 2), width // 2 + 1, dtype=float)
        weights = offsets / np.sum(offsets ** 2)
        padded = np.pad(mfcc, ((0, 0), (width // 2, width // 2)), mode="edge")
        delta = np.stack([np.convolve(row, weights, mode="valid") for row in padded])
        return mfcc, delta

    @staticmethod
    def _gne_envelope_correlation(samples: np.ndarray, sample_rate: int) -> float:
        """Max envelope correlation across 1 kHz bands (direct waveform, no inverse filter)."""
        nyquist = sample_rate / 2
        envelopes = []
        centers = np.arange(500, min(5000, nyquist - 500) + 1, 500)
        for center in centers:
            low, high = max(60.0, center - 500), min(nyquist - 1, center + 500)
            if high <= low:
                continue
            filtered = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos"), samples)
            envelope = np.abs(hilbert(filtered))
            if envelope.std() > 1e-12:
                envelopes.append(envelope)
        correlations = [abs(float(np.corrcoef(a, b)[0, 1])) for i, a in enumerate(envelopes) for j, b in enumerate(envelopes[i + 1:], i + 1) if j - i >= 1]
        correlations = [value for value in correlations if np.isfinite(value)]
        if not correlations:
            raise ValueError("GNE envelope correlations are unavailable for this signal")
        return float(max(correlations))

    def extract(self, audio_bytes: bytes, audio_metadata: dict, *, mode: str, gender_code: int) -> pd.DataFrame:
        if not self.available:
            raise APIError(503, "speech_extractor_dependency_missing", "Install the speech audio dependencies listed in requirements.txt.")
        if mode not in {"sustained_vowel", "reading"}:
            raise APIError(422, "invalid_speech_mode", "mode must be 'sustained_vowel' or 'reading'.")
        if gender_code not in (0, 1):
            raise APIError(422, "invalid_gender_code", "The v2 training schema requires gender_code 0 or 1; it cannot be inferred from audio.")
        audio = self.read_wav(audio_bytes)
        if mode == "sustained_vowel" and len(audio.samples) / audio.sample_rate < 5:
            raise APIError(422, "short_sustained_vowel", "Sustained-vowel mode requires at least 5 seconds of audio.")
        try:
            import parselmouth
            from parselmouth.praat import call

            y, sr = audio.samples, audio.sample_rate
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            pitch_floor, pitch_ceiling = 75.0, 500.0
            point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
            start, end = 0.0, len(y) / sr
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
            report = call([sound, pitch, point_process], "Voice report", start, start,
                          pitch_floor, pitch_ceiling, 1.3, 1.6, 0.03, 0.45)
            def report_value(label: str) -> float:
                match = re.search(rf"^\s*{re.escape(label)}:\s*([-+0-9.Ee]+)", report, re.MULTILINE)
                if not match:
                    raise ValueError(f"Praat voice report omitted {label}")
                return float(match.group(1))
            jitter = [report_value("Jitter (local)"), report_value("Jitter (local, absolute)"),
                      report_value("Jitter (rap)"), report_value("Jitter (ppq5)")]
            shimmer = [report_value("Shimmer (local)"), report_value("Shimmer (local, dB)"),
                       report_value("Shimmer (apq3)"), report_value("Shimmer (apq5)"),
                       report_value("Shimmer (apq11)")]
            f0 = np.asarray(pitch.selected_array["frequency"], dtype=float)
            f0 = f0[np.isfinite(f0) & (f0 > 0)]
            if len(f0) < 32:
                raise ValueError("Too few voiced frames for the acoustic feature set")
            periods = 1.0 / f0
            mfcc, delta = self._mfcc_and_delta(y, sr)
            hnr = []
            for upper in (500, 1500, 2500, 3500, 3800):
                high = min(float(upper), sr / 2 - 1)
                band = sosfiltfilt(butter(4, [60.0, high], btype="bandpass", fs=sr, output="sos"), y)
                hnr_sound = parselmouth.Sound(band, sampling_frequency=sr)
                harmonicity = hnr_sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=pitch_floor, silence_threshold=0.1, periods_per_window=1.0)
                values = np.asarray(harmonicity.values, dtype=float).ravel()
                values = values[np.isfinite(values)]
                if not values.size:
                    raise ValueError(f"HNR band ending at {upper} Hz could not be measured")
                hnr.append(float(np.mean(values)))
            values = [
                gender_code, *jitter, *shimmer, *hnr,
                self._rpde_return_entropy(periods), self._dfa_alpha(periods),
                self._pitch_period_entropy(periods), self._gne_envelope_correlation(y, sr),
                *np.mean(mfcc, axis=1).tolist(), *np.mean(delta, axis=1).tolist(),
            ]
            if len(values) != len(self.features):
                raise ValueError("The extractor produced an incomplete feature vector")
            invalid = [name for name, value in zip(self.features, values) if not np.isfinite(value)]
            if invalid:
                raise ValueError("The signal does not support finite estimates for: " + ", ".join(invalid))
            return pd.DataFrame([dict(zip(self.features, values))], columns=self.features)
        except APIError:
            raise
        except Exception as exc:
            raise APIError(422, "speech_feature_extraction_failed", f"Could not extract the v2 speech feature set: {exc}") from None
