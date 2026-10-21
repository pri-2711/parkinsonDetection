# Experimental raw-audio speech pipeline (v2)

This pipeline is intentionally separate from `speech_svm_tuned.joblib`. The old model expects 45 legacy columns; their extraction recipe is not fully recoverable from the repository. Matching the number or names of columns would not establish compatibility.

## Extraction contract

Input is decoded PCM WAV, mono downmixed by channel averaging, 8–48 kHz. `sustained_vowel` requires at least five seconds; `reading` is a separate mode and requires at least the backend's general minimum duration. The extractor requires a user-supplied dataset gender code because gender cannot be inferred: `0=man`, `1=woman` (the UCI source encoding). It rejects silence, unsupported mode, invalid gender, non-finite signals, and under-supported measurements.

The output is the ordered 45-column `speech_audio_v2` schema in `app_backend/services/acoustic_features.py`:

| Group | v2 definition/settings |
|---|---|
| Gender | Explicit numeric metadata 0 or 1; never inferred. |
| Jitter (4) | Praat `Voice report`, CC point process, F0 bounds 75–500 Hz, period factor 1.3. Local and RAP/PPQ5 are percentages; local absolute jitter is seconds. |
| Shimmer (5) | Praat `Voice report` with the same bounds and explicit pulse settings. Local/APQ values are percentages; local dB is dB. |
| HNR bands (5) | For each upper cutoff 500/1500/2500/3500/3800 Hz, fourth-order Butterworth bandpass from 60 Hz to that cutoff, then Praat autocorrelation harmonicity; mean frame value in dB, 10 ms step. This is a v2 band-limited Praat measure, not the historical VoiceSauce cepstral HNR. |
| RPDE-like (1) | Normalized first-return-time entropy on standardized 3D lag-1 delay vectors; recurrence radius 0.1·√3; exclude temporal neighbors within lag 2; cap at 1,024 evenly sampled voiced periods. This is the pipeline's explicitly named algorithm and is not claimed to reproduce the published PD RPDE implementation. |
| DFA (1) | Cumulative demeaned voiced periods; linear detrending at 12 logarithmic box sizes from 4 to N/4; slope of log fluctuation against log box size. |
| Pitch-period entropy (1) | Normalized Shannon entropy of a 20-bin histogram of voiced F0 periods; a separate v2 descriptor, not the old PPE. |
| GNE-like (1) | Maximum absolute correlation of Hilbert envelopes across fourth-order 1 kHz bandpass filters with centers every 500 Hz from 500 Hz up to Nyquist. It uses the raw acoustic waveform without inverse filtering and is **not** the reference Michaelis GNE. |
| MFCC (13) | Custom deterministic implementation: 25 ms Hamming frames, 10 ms hop, next-power-of-two FFT, power spectrum, 40 HTK mel triangular filters, log energy, orthonormal type-II DCT; take mean over frames for coefficients 0–12. |
| Delta (13) | First temporal regression derivative of each MFCC using width 9 and edge replication; take mean over frames. |

Praat is accessed through Parselmouth, which calls Praat's native algorithms. SciPy implements the explicitly versioned transforms and custom descriptors. Librosa and openSMILE were investigated but are not runtime dependencies: their built-in feature bundles do not match the custom historical schema, and defaults do not resolve that mismatch.

## Model training

There are currently **no labeled raw WAV files** in this repository, so no v2 model is trained or enabled. Training uses a separate manifest with `audio_path, subject_id, mode, gender_code, status`; mode-specific RBF SVMs use a 75/25 subject-disjoint holdout. The script refuses fewer than eight distinct subjects per mode or splits that do not contain both classes. It writes only to the requested output directory.

```powershell
python -m experiments.speech_audio_v2.train path\to\labeled_audio_manifest.csv
```

Metrics from this script are exploratory holdout results, not clinical validation. Do not copy them to the legacy model report. The API runs extraction but returns HTTP 503 `speech_audio_v2_model_not_trained` until the corresponding mode model artifact exists and passes schema checks.

## Legacy compatibility

The existing model remains available as a saved baseline and is never called with v2 features. The UCI descriptor CSV is not raw audio and cannot train a v2 classifier. A mode-specific WAV dataset with subject IDs, labels, and suitable consent/licensing is required before predictions can be enabled.
