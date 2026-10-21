# Parkinson's Detection and Severity Estimation

Research prototype for Parkinson's disease classification from speech, handwriting, and tremor, with separate motor-UPDRS severity regression from telemonitoring data. Includes a Flask backend with user authentication, assessment workflows, SQLite persistence, and a React frontend.

## Overview

This repository contains:

1. **ML Pipeline**: Reproducible data cleaning, feature engineering, model training, and evaluation with subject-disjoint cross-validation
2. **Flask Backend**: REST API for user accounts, assessments, and model inference
3. **React Frontend**: Web interface for authentication, assessment workflows, and results visualization
4. **Model Artifacts**: Trained classifiers and regression models with documented limitations
5. **Documentation**: Architecture, API reference, and feature compatibility status

**Important**: The classification cohorts are independent and unmatched (no participant appears in more than one modality). No patient-level fusion performance has been measured. A tremor-only assessment works from compatible clinician-rated UPDRS inputs; speech and handwriting remain blocked by incompatible available uploads.

## Installation

### Backend Dependencies

```powershell
python -m pip install -r requirements.txt
python -c "import secrets; from pathlib import Path; p=Path('.env'); p.exists() or p.write_text('HOST=127.0.0.1\nPORT=5000\nAPP_SECRET_KEY='+secrets.token_urlsafe(48)+'\nLOG_LEVEL=INFO\n', encoding='utf-8')"
```

The backend reads `APP_SECRET_KEY` from the repository-root `.env` when the Flask app is created. The generated `.env` is ignored by Git; do not share it. Existing accounts and password hashes stay in `data/parkinson_app.db`.

### Frontend Dependencies

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
```

`VITE_API_URL` defaults to `http://127.0.0.1:5000`; only change it when the backend runs elsewhere.

## Running the Application

### Backend Only

```powershell
python -m app_backend
```

Server binds to `127.0.0.1:5000` by default.

### Frontend Only (requires running backend)

```powershell
cd frontend
npm run dev
```

Frontend runs on `http://127.0.0.1:3000`; the Vite proxy forwards `/api` requests to Flask on port 5000.

Open the frontend and register a username, email, and password of at least eight characters, then log in. If you already have an account, use its existing credentials; setup does not reset account passwords.

To test tremor classification, create an assessment from the dashboard, open Tremor, enter seven clinician-rated UPDRS III item 20/21 scores (0–4), and save. The resulting probability appears in results and history. It is a research output, not a diagnosis. Audio/video uploads remain unavailable because they cannot produce the trained feature schemas.

### Both Servers (recommended)

```powershell
# Terminal 1: Backend
python -m app_backend

# Terminal 2: Frontend
cd frontend
npm run dev
```

## Reproducible ML Pipeline

Run from the repository root:

```powershell
python scripts/run_all.py
```

This regenerates cleaned tables and handwriting features from raw data (raw files are never edited), validates and profiles data, performs grouped evaluation, fits final pipelines, and writes reports/artifacts. `python main.py` remains a compatibility entry point.

## Tuning Experiments

Baseline V1 is archived under `archive/Baseline_V1` with a SHA-256 manifest. To run the separate classification tuning experiment:

```powershell
python experiments/regularized_svm_rf/run_experiments.py
```

This tunes SVM and Random Forest separately for speech, handwriting, and tremor with subject-aware nested validation. It writes tuned artifacts to `experiments/regularized_svm_rf/results/artifacts/`. See the experiment directory for full commands, metrics, and limitations.

### Backend API

- `GET /` & `GET /api` - Root service status & modality discovery
- `GET /api/health` - Health check and model readiness
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login (returns signed token)
- `GET /api/auth/me` - Current user profile (authenticated)
- `POST /api/assessments` - Create assessment (authenticated)
- `GET /api/assessments/<id>` - Get assessment details (authenticated)
- `POST /api/assessments/<id>/complete` - Complete assessment and compute late fusion (authenticated)
- `GET /api/assessments/<id>/results` - Get assessment results (authenticated)
- `POST /api/assessments/<id>/tremor/predict` - Tremor classification from 7 UPDRS III items (0–4)
- `POST /api/assessments/<id>/spiral/predict` - Handwriting classification from complete measured tablet telemetry; browser canvas coordinates alone receive a clear 422 response
- `POST /api/assessments/<id>/speech/predict` - Authenticated multipart PCM WAV processing with speech_audio_v2 extraction; returns HTTP 503 until its mode-specific model is trained
- `GET /api/assessments/<id>/report.pdf` - Download official PDF research assessment report (authenticated)
- `GET /api/history/assessments` - List user assessments (authenticated)
- `GET /api/history/completed` - List completed assessments (authenticated)

### Frontend Features

- **Authentication**: Registration, login, logout with signed token management and password show/hide controls
- **Dashboard**: Assessment status cards for tremor, handwriting, and speech tests with direct action buttons
- **Tremor Assessment**: Redesigned accessible discrete 0–4 segmented rating buttons with documented UPDRS III clinical item descriptions
- **Handwriting Test**: Real-time HTML5 canvas with an Archimedean spiral guide and local pointer telemetry (kinematics, stroke timing, and observed browser pressure/tilt). These pixel-based measures are only a prototype and are not sent to the digitizer-trained classifier.
- **Speech Test**: Choose reading or sustained-vowel mode; audio is processed with a separate v2 schema and is not persisted. Classification stays unavailable until a matching raw-audio model is trained.
- **Results & Fusion**: Late fusion probability blending across available modalities with contributing weights
- **PDF Reporting**: Professional ReportLab downloadable PDF reports containing clinical ratings, kinematic metrics, methodology, and research disclaimers
- **Visual Design**: Cohesive darker sage green palette (`#526B4A`, `#3E5438`, `#788B6A`, `#F6F5F0`, `#FFFFFF`, `#29352C`) with responsive cards and accessible controls


## Current Status

### Functional
- ML pipeline with reproducible training and evaluation
- Backend authentication with JWT tokens and SQLite persistence
- User-isolated assessment creation, updates, and completion
- Fusion logic for combining available modality predictions
- Frontend authentication and navigation
- Frontend assessment workflow with status tracking

### Blocked/Prototype
- **Speech prediction**: Praat/SciPy extracts the separate 45-feature `speech_audio_v2` schema for WAV input. No model trained on that schema is present; no prediction is generated and recordings are not persisted. See `experiments/speech_audio_v2/README.md`.
- **Handwriting prediction**: Camera video cannot provide tablet pressure/grip-angle features
- **Tremor prediction**: Works from the seven clinician-rated UPDRS item inputs used in training; it does not infer ratings from self-report or video
- **Severity estimation**: Not deployed due to poor regression performance (negative R²)
- **Assessment completion**: Tremor-only completion works with compatible UPDRS ratings; speech and handwriting remain unavailable and are omitted from fusion

## Project Structure

```text
src/                          Pipeline configuration, modeling, grouped evaluation
scripts/run_all.py            Single complete pipeline entry point
cleaningPipeline/             Deterministic raw-to-clean data transforms
featureEngineering/           Handwriting trajectory aggregation
raw_dataset/                  Original source data (read only)
cleaned_dataset/              Cleaned source tables
models/artifacts/              Joblib model bundles, out-of-fold predictions, run config
experiments/regularized_svm_rf/ Tuning experiment with nested validation
app_backend/                  Flask API with routes, services, repositories
  routes/                     API endpoints (health, auth, assessments, history, etc.)
  services/                   Business logic (auth, classification, fusion, etc.)
  repositories/               Data access layer (user, assessment)
  database.py                 SQLite persistence
  API.md                      API documentation
  FEATURE_COMPATIBILITY.md    Model compatibility status
frontend/                    React + TypeScript + Vite frontend
  src/                       React components, pages, API client
  package.json               Frontend dependencies
  vite.config.ts             Vite configuration with API proxy
reports/
  data_profile/               Dataset profiles
  feature_analysis/           Feature summaries and correlations
  leakage_audit/              Handwriting leakage audit
  model_results/              Per-task metrics and weight comparisons
  FINAL_REPORT.md             Pipeline findings and interpretation
docs/
  architecture.md             System architecture diagram
archive/Baseline_V1/          Immutable baseline version
PROJECT.md                    Concise architecture overview
PROJECT_REPORT.md             Full system documentation
PROGRESS.md                   Implementation progress and blockers
```

## Design

### Classification
- Speech groups by `id`; handwriting groups by `participant_id`; tremor groups by `participant_code`
- Five-fold `StratifiedGroupKFold` for subject-disjoint evaluation
- Calibrated RBF SVM and Random Forest with sigmoid calibration
- SVM/RF ensemble weights selected from OOF predictions
- Decision-level fusion using normalized modality ROC-AUC weights

### Regression
- Telemonitoring groups by `subject_number`
- Five-fold `GroupKFold` for severity estimation
- Ridge and Random Forest compared against group-mean baseline
- Current R² is negative, indicating poor generalization

### Backend
- SQLite database for users and assessments
- JWT authentication with 24-hour token expiry
- User-isolated data access
- Upload validation for audio and video
- CORS enabled for frontend integration

### Frontend
- React 18 with TypeScript
- Vite for development and build
- React Router for navigation
- Protected routes with authentication
- API client with token management
- Sage-green/cream color scheme

## Limitations

- **Small cohorts**: Only 40 handwriting participants, 80 speech participants, 130 tremor participants
- **Unmatched cohorts**: No patient-level fusion performance measured
- **Speech**: Raw audio feature extraction not implemented
- **Handwriting**: Camera video cannot provide tablet features
- **Tremor**: Seven compatible UPDRS item ratings required; no automated sensor extraction
- **Severity**: Poor regression performance (negative R²)
- **Clinical use**: Not validated for clinical diagnosis

## Documentation

- [Architecture](docs/architecture.md) - System architecture diagram
- [Project Overview](PROJECT.md) - Concise architecture and scope
- [Full Report](PROJECT_REPORT.md) - Detailed system documentation
- [Model Explanation](MODEL_EXPLANATION.md) - Plain-English viva guide to models, validation, fusion, and limitations
- [API Reference](app_backend/API.md) - Backend API endpoints
- [Feature Compatibility](app_backend/FEATURE_COMPATIBILITY.md) - Model input requirements
- [Frontend README](frontend/README.md) - Frontend setup and usage

## Tests

```powershell
# Run from the repository root
python -m unittest discover -s tests -v
python -m unittest discover -s app_backend/tests -v
python -m unittest discover -s experiments/regularized_svm_rf -p 'test_*.py' -v

# Frontend (in frontend directory; npm test is not configured)
npm run build
npm run lint
```

The frontend production build succeeds. The lint command currently cannot run because this repository has no ESLint configuration.

## License

This is a research prototype. See individual dataset licenses for data usage terms.
