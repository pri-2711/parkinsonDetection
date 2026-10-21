# Parkinson's Detection Project - System Report

## Problem Statement

Parkinson's disease (PD) is a neurodegenerative disorder that affects movement control. Early detection and severity monitoring are critical for timely intervention. This project aims to develop a research prototype for:

1. **Classification**: Detecting Parkinson's disease from three independent modalities: speech, handwriting, and tremor
2. **Severity Estimation**: Predicting motor UPDRS scores from telemonitoring voice recordings

## Objectives

- Build reproducible ML pipelines for PD classification and severity regression
- Implement subject-disjoint cross-validation to avoid participant leakage
- Develop a Flask backend for assessment workflows and user accounts
- Preserve model artifacts and evaluation metrics for reproducibility
- Document modality-specific input compatibility and known limitations

## Dataset Sources and Schemas

### Speech Dataset
- **Source**: Precomputed acoustic features (raw audio not in repository)
- **Rows**: 240 recordings from 80 participants
- **Columns**: 48 including identifiers (`id`, `recording`, `status`), `gender`, and 44 acoustic descriptors
- **Features**: Jitter variants, Shimmer variants, HNR, RPDE, DFA, PPE, GNE, 13 MFCC values, 13 Delta values
- **Target**: Binary `status` (0=control, 1=PD)
- **Grouping**: `id` for subject-disjoint splits

### Handwriting Dataset
- **Source**: Tablet trajectory data (x, y, z, pressure, grip angle, timestamp, test ID)
- **Rows**: 110 tests from 40 participants (15 control, 25 PD)
- **Features**: 16 aggregated features including velocity, acceleration, jerk, pressure statistics, grip angle variation, duration, pen-up fraction, spatial spread
- **Target**: Binary `label` (0=control, 1=PD)
- **Grouping**: `participant_id` with duplicate signature handling

### Tremor Dataset
- **Source**: UPDRS items 20 and 21 (tremor-specific motor assessments)
- **Rows**: 130 participants (100 control, 30 PD)
- **Features**: 7 numeric UPDRS tremor items
- **Target**: Binary label derived from `participant_code` (starts with "PD" = 1)
- **Grouping**: `participant_code`
- **Inference**: The saved calibrated tremor SVM is available through the authenticated assessment endpoint when all seven clinician-rated item values are supplied in the exact trained schema (integer 0–4).

### Tremor demographics file
- `raw_dataset/UPDRS/tremors_demo/Demographics.csv` is semicolon-delimited, 66 × 9, with participant ID and demographics/clinical descriptors including Hoehn–Yahr stage.
- It has no UPDRS score target; it is not suitable for training the requested tremor UPDRS severity regressor.

### Telemonitoring Dataset
- **Source**: Parkinson's Telemonitoring Dataset (UCI ML Repository)
- **Rows**: 5,875 recordings from 42 subjects
- **Features**: 19 predictors including age, sex, test time, jitter/shimmer, NHR/HNR, RPDE, DFA, PPE
- **Target**: `motor_updrs` (continuous)
- **Grouping**: `subject_number`

## Data Cleaning

### Cleaning Pipeline (`cleaningPipeline/`)
- **speech_cleaning.py**: Reads and validates precomputed speech features CSV
- **handwriting_cleaning.py**: Processes tablet trajectories, preserves point order
- **tremor_cleaning.py**: Extracts UPDRS items 20/21, derives binary label from participant code
- **telemonitoring_cleaning.py**: Filters to voice features, excludes total UPDRS

### Feature Engineering (`featureEngineering/`)
- **handwriting_features.py**: Aggregates trajectories into 16 participant/test features
- **Duplicate handling**: Participants with identical feature signatures are kept in the same fold

## Model Training

### Classification Pipeline (`src/pipeline.py`)

**Preprocessing**:
- Median imputation with `keep_empty_features=True`
- StandardScaler for SVM (handwriting uses no scaling, speech includes scaling in model bundle)
- Sigmoid probability calibration using group-disjoint folds

**Models**:
- **SVM**: RBF kernel with C=1.0
- **Random Forest**: 300 trees, max_depth=8, min_samples_leaf=3, class_weight="balanced"
- **Ensemble**: Weighted blend of SVM and RF probabilities

**Validation**:
- Five-fold `StratifiedGroupKFold` for outer evaluation
- Four-fold nested `StratifiedGroupKFold` for weight selection within training folds
- Weight selection score: mean(ROC-AUC, F1 at threshold 0.5)
- Tie-breaking: selection score, then recall, then closeness to equal weights

**Output and saved comparison**:
- Nested subject-grouped SVM and Random Forest comparisons and calibrated models are saved under `experiments/regularized_svm_rf/results/`.
- The reported outer-fold selected metrics use folds 2–5; fold 1 is reserved for validation-based fusion weights.
- Saved `nested_selected` results (accuracy / precision / recall / F1 / ROC-AUC): speech 0.760 / 0.778 / 0.729 / 0.753 / 0.824; handwriting 0.947 / 0.966 / 0.966 / 0.966 / 0.946; tremor 0.827 / 0.667 / 0.500 / 0.571 / 0.900.
- SVM versus Random Forest ROC-AUC on those rows: speech 0.824 vs 0.813; handwriting 0.932 vs 0.946; tremor 0.900 vs 0.843.

### Regression Pipeline

**Models**:
- **Ridge**: alpha=10.0 with StandardScaler
- **Random Forest**: 300 trees, max_depth=12, min_samples_leaf=5, max_features=0.8
- **Baseline**: Group-mean prediction

**Validation**:
- Five-fold `GroupKFold` (stratification not applicable for regression)
- Metrics: MAE, RMSE, R²

**Output**:
- OOF predictions saved to `models/artifacts/telemonitoring_oof_predictions.csv`
- Models saved to `models/artifacts/telemonitoring_regressors.joblib`

**Verified grouped results** (42 subjects, 5,875 recordings): Ridge MAE 7.656, RMSE 9.089, R² -0.250; Random Forest MAE 8.096, RMSE 9.841, R² -0.466; group-mean baseline MAE 7.187, RMSE 8.316, R² -0.047. Both fitted candidates underperform the baseline, so severity inference is not exposed as reliable.

## Fusion Configuration (`src/fusion.py`)

**Method**: Decision-level probability fusion
- Modality probabilities combined as weighted sum: `P_final = w_speech * P_speech + w_handwriting * P_handwriting + w_tremor * P_tremor`
- Backend weights use the reserved group validation fold AUCs (speech 0.940972, handwriting 0.904762, tremor 0.841667), normalized to 0.350142 / 0.336668 / 0.313190. Their sum is 1.0.
- Missing modalities are omitted; remaining weights are renormalized
- Default classification threshold: 0.5

**Limitation**: The three modality cohorts are independent and unmatched. No patient-level fusion performance has been measured.

## Backend Implementation

### Flask API (`app_backend/`)

**Routes**:
- `GET /`: API root discovery and service summary
- `GET /api`: API status and modality overview
- `GET /api/health`: Health check, model readiness, and diagnostic capability flags
- `POST /api/auth/register`: User registration with password hashing
- `POST /api/auth/login`: User login with signed HMAC-SHA256 token
- `GET /api/auth/me`: Current user profile (authenticated)
- `POST /api/speech/predict`: Speech upload validation and diagnostic response
- `POST /api/spiral/predict`: Spiral/handwriting upload validation
- `POST /api/tremor/submit`: Legacy prototype tremor input
- `POST /api/assessments/<id>/tremor/predict`: Authenticated classifier inference from exact 7 UPDRS III items (0–4)
- `POST /api/assessments/<id>/spiral/predict`: Authenticated handwriting classifier inference from complete measured tablet telemetry; canvas-only coordinates receive HTTP 422
- `POST /api/assessments/<id>/speech/predict`: Authenticated speech reading passage logging and diagnostic 501 response
- `GET /api/assessments/<id>/report.pdf`: Official downloadable PDF research assessment report
- `POST /api/assessments`: Create authenticated assessment
- `GET /api/assessments/<id>`: Get assessment details (authenticated, user-isolated)
- `POST /api/assessments/<id>/complete`: Complete assessment and calculate decision-level late fusion
- `GET /api/assessments/<id>/results`: Get assessment results and fusion details (authenticated)
- `GET /api/history/assessments`: List user assessments (authenticated)
- `GET /api/history/completed`: List completed assessments (authenticated)
- `GET /api/history/assessments/<id>`: Get specific assessment (authenticated)
- `GET /api/results/<id>`: Get assessment results (authenticated)

**Authentication**:
- Passwords hashed with werkzeug PBKDF2
- Signed tokens with HMAC-SHA256, 24-hour expiry
- `login_required` decorator for protected routes
- User-isolated data access via `g.user_id`

**Persistence**:
- SQLite database at `data/parkinson_app.db` (configurable via `DATABASE_PATH`)
- Tables: `users` (id, username, email, pw_hash, timestamps), `assessments` (id, user_id, status, steps, results, timestamps)
- Thread-safe with WAL mode and foreign keys enabled

**Model Loading**:
- Loads tuned artifacts from `experiments/regularized_svm_rf/results/artifacts/`
- Models: `speech_svm_tuned.joblib`, `handwriting_svm_tuned.joblib`, `tremor_svm_tuned.joblib`
- ModelRegistry validates feature order and provides `predict()` interface

### Services (`app_backend/services/`)
- `auth.py`: Registration, login, token management
- `classification_service.py`: Modality prediction wrapper
- `fusion_service.py`: Fusion of modality probabilities
- `severity_service.py`: UPDRS regression (disabled due to negative R²)
- `history_service.py`: Assessment history queries
- `pdf_report.py`: ReportLab PDF assessment report generation
- `sessions.py`: Legacy in-memory session management (retained for compatibility)
- `audio_preprocessing.py`: WAV validation
- `acoustic_features.py`: Acoustic feature extractor interface with 501 diagnostic
- `spiral_preprocessing.py`: Image/video validation
- `trajectory_features.py`: Tablet feature extraction matching training schema
- `tremor_placeholder.py`: Tremor input validation

### Repositories (`app_backend/repositories/`)
- `user_repository.py`: User CRUD operations
- `assessment_repository.py`: Assessment CRUD operations

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ User account and assessment history                           │
│ Private profile · saved tests · progress tracking             │
└──────────────────────────────┬───────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Handwriting  │      │    Speech    │      │    Tremor    │
│ Spiral video │      │  Recorded    │      │ Body location│
│              │      │  paragraph   │      │ + intensity  │
│ Classification│     │ Classification│     │ Classification│
│              │      │ + severity   │      │ + severity   │
└───────┬──────┘      └───────┬──────┘      └───────┬──────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
┌───────────────────────────────┐
│ Classification fusion         │
│ Combine the three modality    │
│ predictions using validated   │
│ weights                       │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Results and history           │
│ Classification outcome ·      │
│ eligible severity estimate ·  │
│ longitudinal trends           │
└───────────────────────────────┘
```

**Note**: Severity estimation is conditional on validated model availability and compatible inputs. The diagram represents the intended architecture where some input/model compatibility blockers may remain (e.g., raw audio feature extraction and camera-to-tablet feature mapping are not yet implemented).

## Limitations

### ML Pipeline
- **Small cohorts**: Only 40 handwriting participants, 80 speech participants, 130 tremor participants
- **Unmatched cohorts**: No participant appears in more than one modality; patient-level fusion not measured
- **Handwriting duplicates**: 7-row control signature and 2-row PD signature shared across participants
- **Regression performance**: Negative R² indicates telemonitoring model does not generalize to unseen subjects

### Backend
- **Speech prediction blocked**: No verified raw-audio feature extractor matching the saved model schema (45 features including gender, MFCC, Delta definitions)
- **Handwriting prediction limited**: The model runs only on compatible measured tablet telemetry. Browser canvas and camera inputs do not provide the required sensor signals; the canvas no longer substitutes made-up pressure or grip-angle values.
- **Tremor prediction**: Available only from the seven clinician-rated UPDRS III item 20/21 values; this research probability is not a clinical diagnosis
- **Severity estimation**: Not deployed due to poor regression performance
- **Assessment completion**: A tremor-only assessment can complete with valid model-generated tremor probability; speech and handwriting remain unavailable and are omitted from fusion.
- **Authentication**: Basic JWT without refresh tokens or multi-factor auth

### General
- The React/TypeScript frontend production build succeeds with the installed lockfile dependencies. `npm run lint` is currently blocked because the repository has no ESLint configuration.
- No external clinical validation
- Models are research prototypes, not clinical diagnostic tools

## Testing

### ML Tests (`tests/`)
- `test_fusion.py`: Fusion probability combination with weight renormalization
- Archive verification: `archive/Baseline_V1/verify_archive.py`

### Backend Tests (`app_backend/tests/`)
- `test_backend.py`: API health, model loading, upload validation, session state machine, tablet feature extraction
- `test_assessment_db.py`: authenticated tremor inference, exact input validation, and probability persistence

### Running Tests
```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s app_backend/tests -v
python -m unittest discover -s experiments/regularized_svm_rf -p 'test_*.py' -v
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip

### Installation
```powershell
python -m pip install -r requirements.txt
python -c "import secrets; from pathlib import Path; p=Path('.env'); p.exists() or p.write_text('HOST=127.0.0.1\\nPORT=5000\\nAPP_SECRET_KEY='+secrets.token_urlsafe(48)+'\\nLOG_LEVEL=INFO\\n', encoding='utf-8')"
```

### Running the Backend
```powershell
python -m app_backend
```
Server binds to `127.0.0.1:5000` by default.

### Running the ML Pipeline
```powershell
python scripts/run_all.py
```
This regenerates cleaned data, trains models, and generates reports.

### Running Tuning Experiments
```powershell
python experiments/regularized_svm_rf/run_experiments.py
```

## Configuration

### Environment Variables (`.env`)
- `HOST`, `PORT`: Server binding
- `APP_SECRET_KEY`: JWT signing secret (required for auth)
- `LOG_LEVEL`: Logging verbosity
- `MODEL_ARTIFACT_DIR`: Path to tuned model artifacts
- `DATABASE_PATH`: SQLite database path
- `MAX_REQUEST_BYTES`, `AUDIO_MAX_BYTES`, `SPIRAL_MAX_BYTES`: Upload size limits
- `AUDIO_MIN_SECONDS`, `AUDIO_MAX_SECONDS`, `SPIRAL_MAX_SECONDS`: Duration limits
- `SESSION_TTL_SECONDS`: In-memory session TTL
- `TREMOR_ALLOWED_LOCATIONS`, `TREMOR_MAX_INTENSITY`: Tremor input constraints

## Future Work

### High Priority
- Implement verified raw-audio feature extraction for speech with gender input contract
- Investigate computer-vision approaches for camera-to-tablet feature mapping
- Improve telemonitoring regression with better features or regularization
- Add refresh tokens and session management improvements

### Medium Priority
- External validation on larger, matched cohorts
- Patient-level fusion evaluation when matched data is available
- Extend automated frontend coverage; the production build currently verifies TypeScript compilation and bundling
- Severity estimation from speech/handwriting when compatible models exist

### Low Priority
- Additional modalities (gait, facial expressions)
- Time-series analysis for longitudinal monitoring
- Model interpretability and explanation features
- Integration with clinical systems

## Conclusion

This project provides a reproducible research prototype for Parkinson's disease classification and separate severity estimation. Its authenticated Flask workflow can run a genuine tremor classifier from the seven compatible clinician-rated UPDRS inputs, persist its probability, and complete a tremor-only assessment. Speech and camera-based handwriting inference remain blocked by feature incompatibilities, and severity is not deployed because grouped validation underperformed the baseline. The frontend production build and 31 Python tests pass; lint is blocked by missing ESLint configuration. The system is for research, not clinical diagnosis.
