# Research Prototype Audit: Parkinson's Detection & Severity Estimation

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-13  
**Repository Branch:** main (incorporating model_v2 findings)  
**Primary Language/Frameworks:** Python 3.12 (Flask backend), TypeScript (React 18 + Vite frontend), SQLite 3 (Database)

---

## 1. Executive Summary

This audit assesses the state of the final-year CSE research project investigating multimodal Parkinson's disease classification (speech, handwriting, tremor) and telemonitoring motor-UPDRS regression. The application is strictly a **research prototype** designed for academic experimentation, not a medical device or diagnostic instrument.

---

## 2. Component Status Classification

| Modality / Component | Baseline Status | Implementation State | Production Artifacts | Primary Blocker / Limitation |
|---|---|---|---|---|
| **Tremor Classification** | Working | Fully Integrated | `tremor_svm_tuned.joblib` | Requires clinician or standardized 0–4 UPDRS ratings. |
| **Handwriting (Tablet Trajectory)** | Partially Functional | Fully Integrated | `handwriting_svm_tuned.joblib` | Requires coordinate telemetry; ordinary camera cannot directly measure pressure/grip angle. Interactive canvas & CSV trajectory supported. |
| **Handwriting (Camera/Video)** | Blocked | Experimental CV Inspection | OpenCV video processor | Video cannot measure tablet pressure/grip angle; visual path irregularity and tremor oscillations are visual proxies only. |
| **Speech Classification** | Blocked | Audio Validation & Diagnostic | `speech_svm_tuned.joblib` | Model expects 45 precomputed acoustic features (jitter, shimmer, HNR, RPDE, DFA, PPE, gender). No verified raw-audio extractor exists in the repo matching this schema. |
| **Telemonitoring Regression** | Non-generalizable | Disabled / Experimental | `telemonitoring_regressors.joblib` | Negative R² (-0.250 Ridge, -0.466 RF) under participant-independent GroupKFold cross-validation; worse than group mean. |
| **Decision-Level Fusion** | Validated | Fully Integrated | `fusion_validation_weights.json` | Weights (Speech: 0.3501, Handwriting: 0.3367, Tremor: 0.3132) estimated on reserved fold 1. Patient cohorts are unmatched; patient-level multimodal accuracy is not claimed. |
| **Authentication & DB** | Working | Fully Integrated | `data/parkinson_app.db` | Custom HMAC-SHA256 JWT auth, user ownership isolation, SQLite persistence. |
| **PDF Reporting** | Missing | Implemented | ReportLab service | Generates professional research assessment reports with disclaimers. |

---

## 3. Detailed Audit of Classification Modalities

### 3.1 Speech Classification
- **Dataset:** `cleaned_dataset/speech_clean.csv` (1,040 rows from 80 subjects: 40 PD, 40 Control).
- **Features:** 45 features including `gender`, `jitter_rel`, `jitter_abs`, `jitter_rap`, `jitter_ppq`, `shim_loc`, `shim_db`, `shim_apq3`, `shim_apq5`, `shi_apq11`, etc.
- **Trained Model:** `experiments/regularized_svm_rf/results/artifacts/speech_svm_tuned.joblib` (Calibrated RBF SVM, $C=1.0$, $\gamma=0.01$).
- **Verified Metrics (Outer Folds 2–5):**
  - Accuracy: $0.7604 \pm 0.0269$
  - Precision: $0.7778 \pm 0.0271$
  - Recall: $0.7292 \pm 0.0538$
  - F1-Score: $0.7527 \pm 0.0326$
  - ROC-AUC: $0.8238 \pm 0.0321$
- **Backend Integration:** Model loads successfully via `ModelAdapter`. Accepts 45 precomputed tabular features.
- **Blocker:** Incoming WAV files cannot be converted to the exact 45 features because training used precomputed tabular values without publishing the raw-audio extraction code. Live inference returns HTTP 501 `raw_audio_feature_extractor_unavailable` rather than faking outputs.

### 3.2 Handwriting / Spiral Classification
- **Dataset:** `cleaned_dataset/handwriting_clean.csv` (PaHaW tablet drawing telemetry, 75 subjects).
- **Features:** 16 tabular kinematic and spatial features (`n_points`, `path_length`, `mean_velocity`, `max_velocity`, `mean_acceleration`, `max_acceleration`, `mean_jerk`, `max_jerk`, `pressure_mean`, `pressure_std`, `pressure_range`, `grip_angle_std`, `movement_duration`, `pen_up_fraction`, `spatial_std`, `bounding_box_diagonal`).
- **Trained Model:** `experiments/regularized_svm_rf/results/artifacts/handwriting_svm_tuned.joblib` (Calibrated RBF SVM, $C=0.1$, $\gamma=0.01$).
- **Verified Metrics (Outer Folds 2–5):**
  - Accuracy: $0.9467 \pm 0.0625$
  - Precision: $0.9661 \pm 0.0625$
  - Recall: $0.9661 \pm 0.0667$
  - F1-Score: $0.9661 \pm 0.0399$
  - ROC-AUC: $0.9322 \pm 0.0625$
- **Backend Integration:** `app_backend/services/trajectory_features.py` executes the exact training feature extractor on tablet trajectory data `(point_index, x, y, z, pressure, grip_angle, timestamp, test_id)` and produces identical feature schemas.
- **Input Strategy:** 
  - Real-time HTML5 PointerEvents Canvas drawing (capturing coordinates, timestamp, and pressure/tilt where available from stylus or pointer devices).
  - Tablet CSV telemetry upload.
  - Video upload evaluated with a separate OpenCV visual inspection pipeline (deviation from Archimedean spiral, velocity smoothness, contour irregularities) and annotated frame output, clearly labeled as experimental visual proxy inspection.

### 3.3 Tremor Classification
- **Dataset:** `cleaned_dataset/tremor_clean.csv` (260 clinical evaluations).
- **Features:** 7 clinician-rated UPDRS III items:
  1. Item 20: Tremor at rest — Head (face, lips, chin)
  2. Item 20: Tremor at rest — Right Upper Extremity (RUE)
  3. Item 20: Tremor at rest — Left Upper Extremity (LUE)
  4. Item 20: Tremor at rest — Right Lower Extremity (RLE)
  5. Item 20: Tremor at rest — Left Lower Extremity (LLE)
  6. Item 21: Action/postural tremor — Right Upper Extremity (RUE)
  7. Item 21: Action/postural tremor — Left Upper Extremity (LUE)
- **Trained Model:** `experiments/regularized_svm_rf/results/artifacts/tremor_svm_tuned.joblib` (Calibrated RBF SVM, $C=1.0$, $\gamma=0.1$).
- **Verified Metrics (Outer Folds 2–5):**
  - Accuracy: $0.8269 \pm 0.0222$
  - Precision: $0.6667 \pm 0.0614$
  - Recall: $0.5000 \pm 0.1361$
  - F1-Score: $0.5714 \pm 0.0939$
  - ROC-AUC: $0.8997 \pm 0.0216$
- **Backend Integration:** Fully operational at `POST /api/assessments/<id>/tremor/predict`. Server strictly validates integer ratings $\in [0, 4]$.

---

## 4. Audit of Telemonitoring Regression (Severity Estimation)

- **Dataset:** `cleaned_dataset/telemonitoring_clean.csv` (Oxford Parkinson's Disease Telemonitoring Dataset).
- **Observations:** 5,875 recordings across 42 participants.
- **Target:** `motor_updrs` (continuous score 5.0–39.5).
- **Input Features:** 19 voice acoustic features (`age`, `sex`, `test_time`, 5 jitter features, 6 shimmer features, NHR, HNR, RPDE, DFA, PPE). Excludes `subject_number` and `total_updrs`.
- **Validation Strategy:** 5-fold `GroupKFold` grouped by `subject_number`.
- **Verified Evaluation Scores:**
  - **Ridge Regression ($\alpha=10$):** $\text{MAE} = 7.656$, $\text{RMSE} = 9.089$, $R^2 = -0.250$
  - **Random Forest Regressor:** $\text{MAE} = 8.096$, $\text{RMSE} = 9.841$, $R^2 = -0.466$
  - **Group Mean Baseline:** $\text{MAE} = 7.187$, $\text{RMSE} = 8.316$, $R^2 = -0.047$
- **Investigation of Potential Second Regression Dataset:**
  - `raw_dataset/UPDRS/` contains only `telemonitoring/` and `tremors_demo/` (which is tabular classification data).
  - No other continuous UPDRS regression pipeline exists in the repository.
- **Architectural Decision:** Because both regression models yield negative $R^2$ (predicting worse than the sample mean for unseen participants), severity regression is **disabled** in the clinical assessment workflow and explicitly documented as a known research limitation.

---

## 5. Decision-Level Fusion Audit

- **Approach:** AUC-normalized weighted probability blend:
  $$P_{\text{fused}} = \frac{\sum_{m \in M_{\text{available}}} w_m P_m}{\sum_{m \in M_{\text{available}}} w_m}$$
- **Weights:** Derived from reserved fold 1 validation ROC-AUC:
  - Speech weight: $0.350142$ ($\text{AUC} = 0.940972$)
  - Handwriting weight: $0.336668$ ($\text{AUC} = 0.904762$)
  - Tremor weight: $0.313190$ ($\text{AUC} = 0.841667$)
- **Missing Modality Handling:** Renormalizes available weights dynamically. If only one modality is completed, displays single-modality prediction.
- **Multimodal Evaluation Disclaimer:** Because speech, handwriting, and tremor cohorts are drawn from independent participant populations, fused predictions do not represent empirical patient-level multimodal accuracy.

---

## 6. Backend and Security Audit

- **Framework:** Flask with application factory (`create_app`).
- **Database:** SQLite (`data/parkinson_app.db`) with `users` (22 registered accounts preserved) and `assessments` tables.
- **Authentication:** Custom HMAC-SHA256 token verification; passwords hashed using standard salted key derivation.
- **Isolation:** Users can only query and mutate their own assessments (`WHERE user_id = ?`).
- **File Upload Security:** Enforces size limits (`AUDIO_MAX_BYTES = 5MB`, `SPIRAL_MAX_BYTES = 25MB`), MIME type checking, magic bytes verification, temporary file cleanup.
- **Endpoints:**
  - `GET /` and `GET /api`: API discovery and metadata.
  - `GET /api/health`: Service health and model readiness check.
  - `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`: User management.
  - `POST /api/assessments`, `GET /api/assessments/<id>`, `POST /api/assessments/<id>/complete`: Assessment lifecycle.
  - `POST /api/assessments/<id>/tremor/predict`: UPDRS tremor inference.
  - `POST /api/assessments/<id>/spiral/predict`: Canvas trajectory or tablet CSV inference.
  - `POST /api/assessments/<id>/spiral/video`: Video computer vision analysis.
  - `POST /api/assessments/<id>/speech/predict`: Speech validation and audio analysis.
  - `GET /api/assessments/<id>/report.pdf`: PDF report generation.
  - `GET /api/history/assessments`, `GET /api/history/completed`: Longitudinal history.
