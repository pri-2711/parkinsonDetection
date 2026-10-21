# Master Implementation Pass — Verified Progress and Audit Report

## Raw-audio feature pipeline (2026-10-13)

- Added a separate `speech_audio_v2` PCM WAV pipeline using Parselmouth/Praat plus explicit SciPy signal processing. It extracts the 45-column versioned schema: Praat jitter/shimmer, band-limited Praat HNR, explicitly defined RPDE-like return entropy, DFA slope, pitch-period entropy, custom envelope-correlation GNE-like measure, 13 MFCC means, 13 MFCC-delta means, and required user-supplied gender code. Recording mode is explicit (`sustained_vowel` or `reading`); sustained vowels require at least five seconds.
- Legacy compatibility check: the v2 ordered column names differ from the saved model schema, despite both containing 45 columns. The legacy SVM is never called with v2 values. The exact legacy RPDE/PPE/GNE, HNR and MFCC recipes remain unrecoverable. V2 formulas/settings and differences from the historic definitions are recorded in `experiments/speech_audio_v2/README.md`.
- Added a mode-specific, subject-disjoint SVM training pipeline at `experiments/speech_audio_v2/train.py`. The repository has no labeled raw WAV files, so no v2 model was trained. The upload routes now complete feature extraction and return HTTP 503 `speech_audio_v2_model_not_trained`, with no prediction; an assessment records only extraction status/metadata, never the audio or feature values.
- Installed Parselmouth 0.4.7, Librosa 1.0.0, and openSMILE 2.6.0 for investigation. Runtime requirements include Parselmouth and SoundFile; Librosa/openSMILE are not imported because their standard feature sets do not resolve the historical compatibility gap. Pip initially changed NumPy 2.5.4 to 2.2.6; NumPy 2.5.4 was restored and the full tests passed afterward.
- Frontend now lets the user choose reading vs sustained-vowel mode and explicitly supplies the dataset gender code; it explains that no model is trained for the new schema. Audio is processed in memory and not retained.
- Verification: `python -m unittest discover -s app_backend/tests -v` passed all 23 tests; synthetic five-second PCM voice generated finite 45-feature output in both modes, and authenticated upload reached extraction then returned the expected 503. Frontend build, trainer `--help`, Python compilation, and `git diff --check` passed. Synthetic extraction is a processing test, not clinical validation. No repository WAV recording exists for real-audio evaluation.

## Speech audio transport checkpoint (2026-10-13; superseded by the v2 pipeline above)

- Fixed the actual frontend gap: authenticated speech submissions now attach the selected/recorded audio as multipart field `file`. Browser recordings are decoded and encoded to mono PCM16 WAV before upload; uploaded WAVs are passed through unchanged. The backend validates audio and assessment ownership, records format/duration/sample-rate metadata with `audio_received: true`, and explicitly records `audio_stored: false`; no bytes are written to the database or artifacts.
- The endpoint intentionally remains HTTP 501. The saved model expects 45 ordered columns, but there are no raw training WAVs locally for paired feature verification. The original dataset task is a 5-second sustained /a/ recorded at 44.1 kHz/16-bit, whereas the app captures a 15–20 second reading passage. UCI identifies gender as 0=man, 1=woman. The original study describes 80 ms autocorrelation-based pitch periods, waveform matching for jitter, VoiceSauce cepstral HNR in five bands averaged over voiced frames, plus MFCC/deltas and nonlinear features, but the available implementation details do not resolve all extraction settings or units. Parselmouth, Librosa and openSMILE were not installed; no framework's default feature set can be assumed compatible. No package was installed because no source waveform/reference extractor exists to verify the outputs, and none can exactly establish the app's different reading-task compatibility.
- Files changed for this checkpoint: `frontend/src/api/client.ts`, `frontend/src/api/assessments.ts`, `frontend/src/pages/Speech.tsx`, `app_backend/routes/assessment.py`, `app_backend/tests/test_assessment_db.py`, `app_backend/FEATURE_COMPATIBILITY.md`, `app_backend/API.md`, `README.md`, and this handover.
- Historical verification: backend passed 21 tests at this checkpoint; current 23-test results are above. Browser microphone/device testing has not been performed.
- Current next step: obtain labeled raw audio with subject IDs separately for each mode, run the documented trainer, verify subject-level performance, then confirm v2 artifacts load and a live browser request yields the trained model's actual output.

## Focused compatibility and canvas telemetry pass (2026-10-13)

- Audited the loaded speech bundle against the raw/cleaned CSV and training wrapper. Its 45 ordered inputs are gender; four jitter, five shimmer, five HNR, RPDE/DFA/PPE/GNE, 13 MFCC, and 13 Delta columns. The cleaning script only normalizes an existing feature table. No repository or installed Parselmouth, Librosa, or openSMILE extractor establishes the original calculations. Exact pitch/window settings, units, MFCC/Delta recipe, nonlinear feature implementations, HNR suffix semantics, and gender encoding remain unresolved. Kept the raw-audio route at HTTP 501 and did not feed any approximate values into the saved baseline.
- Upgraded the existing spiral canvas with a local telemetry proof of concept: elapsed time starts on first pointer-down; samples retain coordinates, stroke IDs/boundaries, pointer source, browser-reported pressure/tilt when observed, and derived pixel distance/speed/acceleration/jerk; pause intervals and pressure coverage are summarized. Ink width uses reported pressure where available and fixed-width ink otherwise. The feature coverage UI labels canvas kinematics as approximate and grip angle/compatible pen-up fraction unavailable. The page no longer posts canvas-only inputs to the tablet model; no prediction or assessment result is generated or persisted from this prototype.
- Files changed in this pass: `app_backend/FEATURE_COMPATIBILITY.md`, `app_backend/tests/test_backend.py`, `frontend/src/pages/Handwriting.tsx`, `frontend/src/pages/Results.tsx`, `README.md`, `MODEL_EXPLANATION.md`, and new `frontend/src/features/handwritingTelemetry.js`, `.d.ts`, and `.test.mjs`.
- Verification: `python -m unittest discover -s app_backend/tests -v` passed 21 tests, including exact 45-column saved-model/raw-table order and the existing WAV HTTP 501 behavior. `node --test src/features/handwritingTelemetry.test.mjs` passed 2 tests for kinematics/stroke gaps and sensor availability. `npm.cmd run build` passed (TypeScript and Vite, 49 modules). `git diff --check` passed. First telemetry test run exposed an incorrect expected pause count in the test; the expected result was corrected to reflect both observed 300 ms gaps, then both telemetry tests passed.
- Not tested: physical stylus/browser behavior or full browser interaction. Parselmouth, Librosa, and openSMILE are not installed; no package was installed. No models were trained/replaced, no video stage started, and no database or saved model artifact was changed.
- Next step before enabling speech inference: obtain the original speech feature-generation code/specification and gender mapping, or approve a separate documented feature schema and retraining/validation. Next canvas step: test on target mouse/touch/pressure-pen devices, verify hardware-reported telemetry, then collect labeled canvas examples before considering a separate classifier. Do not connect this prototype to the existing tablet model.

## Final repair pass: stale backend, truthful spiral input, and viva documentation

### Changes completed

- Identified two Flask `python -m app_backend` processes listening on port 5000. The older process served old route code (its root API payload still included the old CSE service label and prior logs showed 404s); the current source and a fresh app registered `GET /`, `GET /api`, speech and spiral routes. Stopped the stale process and confirmed the sole current server is serving the updated root/health routes. CORS preflight for the authenticated spiral request returned 200, so CORS was not the underlying stale-route cause.
- Improved API client network/PDF errors and routed authenticated PDF downloads through the configured backend URL. Speech page now keeps the intentional 501 message visible instead of navigating away. It explicitly says audio is only previewed locally and is not uploaded/stored by that page.
- Removed the CSV and video upload modes from the handwriting page. The canvas remains with its Archimedean guide, pointer-based coordinate capture, monotonic elapsed timing, clear action, and submit button. It no longer invents pressure or grip angle. The spiral API now rejects missing measured tablet sensor fields with HTTP 422 `tablet_telemetry_required`; compatible tablet inputs still use the shared training extractor and fitted model.
- Updated health/readiness, feature compatibility, API and project docs so tablet classification is distinguished from unsupported canvas/camera input. Removed CSE wording from root API/PDF labels; preserved the product title, sage styling, and research disclaimers.
- Added root `MODEL_EXPLANATION.md`, linked from README. Its classification/regression metrics, fusion weights, protocols, and caveats were checked against experiment JSON/Markdown, profiles, feature extractors, and source modules.

### Verification

- `python -m unittest discover -s app_backend/tests -v`: 21 passed, including a real authenticated handwriting route call using a complete measured trace from the repository dataset (saved model inference, persistence, fusion, and PDF), a canvas-only missing-sensor rejection, intentional speech 501, tremor persistence, history, and PDF download.
- `python -m unittest discover -s tests -v`: 10 passed.
- `python -m unittest discover -s experiments/regularized_svm_rf -p 'test_*.py' -v`: 4 passed.
- `npm.cmd run build` in `frontend/`: passed (TypeScript and Vite production build, 48 modules).
- Live HTTP check: `/`, `/api`, and `/api/health` return 200 on the refreshed server; unauthenticated spiral/speech assessment paths return registered-route 401 `missing_token`; spiral CORS preflight returns 200 with the frontend origin. The Vite page and `/api/health` proxy both returned 200 after retry (one initial short-timeout proxy probe timed out while the server was starting).
- The runtime live server is HTTP-tested, but no browser automation/E2E run was performed. Existing production SQLite data and saved model files were not intentionally changed.
- Final read-only SQLite count: 22 users and 2 assessments. These differ from an earlier handover checkpoint (22 users, 0 assessments); no production account or assessment was deleted or reset. All test-created users/assessments used temporary test databases.

### Remaining input limitations

- The browser canvas does not contain the measured pressure/grip-angle units needed by the handwriting model, so its submit action receives the explicit 422 response. A compatible digitizer payload is required for a genuine spiral prediction.
- Speech has no verified extractor for the saved 45-column schema and returns intentional HTTP 501; no speech probability is generated. Severity regression remains disabled as described in the research report.

**Date:** 2026-10-13  
**Status:** Comprehensive Audit, Repair, Integration, PDF Generation, UI Redesign, and Verification Complete  
**Working Tree Status:** All 33 Python unit tests passing; React/Vite production build passing; 0 destructive rewrites; production DB preserved.

---

## 1. Master Implementation Summary

### A. Repository Audit & Baseline
- Created `AUDIT.md` providing a comprehensive audit of all components, modalities, models, datasets, and integration gaps.
- Confirmed working features: Tremor classification (calibrated SVM), Handwriting trajectory classification (calibrated SVM on tablet telemetry), Late decision fusion, Authentication (HMAC-SHA256 JWT), SQLite persistence, PDF report generation.
- Documented genuine research limitations: Speech live audio extraction requires a 45-feature acoustic pipeline matching training schemas; ordinary video cannot measure tablet pressure/grip angle; telemonitoring motor-UPDRS regression yields negative $R^2$ (-0.250 Ridge, -0.466 RF) and remains disabled for clinical severity estimation.
- Preserved all 22 registered user accounts and assessment history in `data/parkinson_app.db`. Maintained safe `.env.example` and Git-ignored `.env`.

### B. Model Training, Evaluation, and Verification
All models were verified from saved artifacts in `experiments/regularized_svm_rf/results/artifacts/` and evaluated on outer participant-held-out folds 2–5 (with fold 1 reserved for validation fusion weights):

| Modality | Model | Exact Artifact | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Speech** | Tuned Calibrated SVM | `speech_svm_tuned.joblib` | 45 | 0.7604 | 0.778 | 0.729 | 0.753 | 0.824 |
| **Handwriting** | Tuned Calibrated SVM | `handwriting_svm_tuned.joblib` | 16 | 0.9467 | 0.966 | 0.966 | 0.966 | 0.932 |
| **Tremor** | Tuned Calibrated SVM | `tremor_svm_tuned.joblib` | 7 | 0.8269 | 0.667 | 0.500 | 0.571 | 0.900 |

#### Fusion Weights (Derived from Reserved Validation Fold 1)
- Speech: $0.350142$ ($\text{AUC} = 0.940972$)
- Handwriting: $0.336668$ ($\text{AUC} = 0.904762$)
- Tremor: $0.313190$ ($\text{AUC} = 0.841667$)

#### Telemonitoring Motor-UPDRS Severity Regression
- **Target:** `motor_updrs` on 42 participants (5,875 recordings).
- **Ridge ($\alpha=10$):** $\text{MAE} = 7.656$, $\text{RMSE} = 9.089$, $R^2 = -0.250$
- **Random Forest:** $\text{MAE} = 8.096$, $\text{RMSE} = 9.841$, $R^2 = -0.466$
- **Baseline (Group Mean):** $\text{MAE} = 7.187$, $\text{RMSE} = 8.316$, $R^2 = -0.047$
- **Decision:** Because $R^2 < 0$, the model performs worse than the sample mean for unseen subjects. Severity estimation is explicitly disabled in the application and documented as an academic research limitation.

---

## 2. Infrastructure & Backend Implementation

1. **Root API & Discovery:**
   - Implemented `GET /` and `GET /api` in `app_backend/routes/health.py` returning service identification, status, supported modalities, and disclaimers.
   - Enhanced `GET /api/health` with diagnostic model readiness details.
2. **Professional PDF Reports (Phase 9):**
   - Implemented `app_backend/services/pdf_report.py` using `reportlab`.
   - Generates official downloadable PDF assessment reports containing:
     - Unique assessment ID, timestamp, anonymous participant reference.
     - Prominent research prototype disclaimer box.
     - Detailed 7-item UPDRS tremor ratings breakdown with clinical item labels.
     - Handwriting trajectory / kinematics metrics and model outcome.
     - Speech assessment reading passage and feature extraction status.
     - Decision-level late fusion composite score and contributing weights.
     - Concise technical methodology summary.
   - Added endpoint `GET /api/assessments/<int:assessment_id>/report.pdf` (authenticated).
3. **Handwriting & Spiral Assessment Endpoint (Phase 7):**
   - Added authenticated endpoint `POST /api/assessments/<int:assessment_id>/spiral/predict`.
   - Ingests trajectory points `(point_index, x, y, z, pressure, grip_angle, timestamp, test_id)`.
   - Converts to DataFrame and calls `extract_tablet_trajectory_features(df, features)`.
   - Executes real model inference with `handwriting_svm_tuned.joblib`.
   - Marks `model_verified = True` and updates database step.
4. **Speech Assessment Endpoint (Phase 6):**
   - Added authenticated endpoint `POST /api/assessments/<int:assessment_id>/speech/predict`.
   - Logs standardized reading passage (v1.0), duration, sample rate.
   - Returns transparent HTTP 501 `raw_audio_feature_extractor_unavailable` without faking probabilities, and persists step status.
5. **Tremor Endpoint Enhancement (Phase 5):**
   - Stores `outcome["input_features"]` in database so user entered UPDRS scores are persisted and visible in reports.

---

## 3. Frontend Redesign & Implementation

1. **Darker Sage Theme (Phase 10):**
   - Implemented tokens in `frontend/src/styles/global.css`:
     - Primary deep sage green: `#526B4A`
     - Dark green for active/hover: `#3E5438`
     - Muted medium sage: `#788B6A`
     - Warm off-white background: `#F6F5F0`
     - Clean white cards: `#FFFFFF`
     - Dark readable text: `#29352C`
     - Subtle border: `#D0D7C9`
   - Added styling for segmented buttons, rating badges, canvas drawing, accessible focus states.
2. **Tremor Form Redesign (Phase 5):**
   - Completely replaced dropdowns with discrete 0–4 segmented rating buttons with numeric tick badges.
   - Displays documented UPDRS III clinical descriptions (0: None, 1: Slight, 2: Mild, 3: Moderate, 4: Marked).
   - Validates all 7 fields before submission with keyboard navigation support.
3. **Speech Assessment Page (Phase 6):**
   - Added exact required introductory paragraph.
   - Added standardized reading passage text ("Today is a pleasant day...").
   - Implemented in-browser microphone recording via MediaRecorder API with duration counter and audio preview player.
   - Alternative WAV file upload option.
4. **Handwriting Assessment Page (Phase 7):**
   - Implemented HTML5 interactive canvas drawing pad with Archimedean spiral guide overlay capturing real-time pointer coordinates and pressure.
   - Tablet CSV telemetry upload option.
   - Video upload option with OpenCV visual analysis explanation.
5. **Results & History Pages:**
   - Added "Download Official Research PDF Report" button on both Results and History pages.
   - Interactive buttons to navigate directly to completed assessments.
   - Clear display of contributing fusion modalities and disclaimers.

---

## 4. Verification & Testing

1. **Backend Tests:**
   - `python -m unittest discover -s app_backend/tests -v`: 19 passed.
   - `python -m unittest discover -s tests -v`: 10 passed.
   - `python -m unittest discover -s experiments/regularized_svm_rf -p 'test_*.py' -v`: 4 passed.
   - **Total:** 33 unit tests passed cleanly.
2. **Frontend Build:**
   - `npm run build` in `frontend/`: Passed cleanly in 1.20s with 0 errors.
3. **Database Integrity:**
   - Verified `data/parkinson_app.db` contains 22 user accounts intact.

---

## Current integrated lifecycle checkpoint (2026-10-13)

This checkpoint supersedes older feature/status claims above where they conflict. No production database migration, reset, or write was performed in this work session.

- The authenticated assessment workflow now saves speech-audio-v2 features and recording metadata without saving raw audio. Until a matching recording-mode model is trained, extraction is reported as `unavailable` for prediction; the legacy 45-feature speech model is not called with v2 values.
- Canvas coordinate and timing telemetry can be saved as a separate observation. The digitizer-trained handwriting classifier is not called with canvas measurements. Camera video remains a separate experimental OpenCV sidecar, with the original and annotated files kept private and associated to the assessment.
- History now lists saved in-progress as well as completed assessments. Results and PDF output distinguish model predictions from saved features and experimental observations. Multi-modality fusion is labeled as an AUC-weighted decision-level score, not a patient-level calibrated probability; unavailable modalities are omitted and active weights renormalized.
- Verified classifier paths: tremor UPDRS inputs produce a genuine saved-model prediction; the handwriting model can predict from compatible digitizer telemetry, which the ordinary browser canvas does not supply. No speech-audio-v2 prediction model is currently available. The application does not claim patient-level multimodal validation.
- Verification on this checkpoint: `python -m unittest discover -s app_backend/tests -q` (27 passed); `python -m unittest discover -s tests -q` (10 passed); `node --test src/features/*.test.mjs` (2 passed); `npm.cmd run build` (passed); Python source compilation (42 files passed); `git diff --check` (passed with line-ending notices only). Synthetic video tests produced an MP4 that OpenCV reopened and decoded, verified authenticated private retrieval and generated a PDF. These are software tests, not participant validation.
- Remaining blockers: labeled raw speech recordings plus a trained/validated mode-specific model; hardware compatible with the handwriting model's digitizer features; matched-cohort evaluation before claiming patient-level fusion performance; real participant-video validation under varied capture conditions. No browser-driven end-to-end test was run.
