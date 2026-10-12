# Parkinson's Detection and Severity Estimation

Research prototype for Parkinson's disease classification from speech, handwriting, and tremor, with separate motor-UPDRS severity regression from telemonitoring data. Includes a Flask backend with user authentication, assessment workflows, SQLite persistence, and a React frontend.

## Overview

This repository contains:

1. **ML Pipeline**: Reproducible data cleaning, feature engineering, model training, and evaluation with subject-disjoint cross-validation
2. **Flask Backend**: REST API for user accounts, assessments, and model inference
3. **React Frontend**: Web interface for authentication, assessment workflows, and results visualization
4. **Model Artifacts**: Trained classifiers and regression models with documented limitations
5. **Documentation**: Architecture, API reference, and feature compatibility status

**Important**: The classification cohorts are independent and unmatched (no participant appears in more than one modality). No patient-level fusion performance has been measured. End-to-end assessment is currently blocked by missing feature extractors for speech and handwriting.

## Installation

### Backend Dependencies

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env with APP_SECRET_KEY and other settings
```

### Frontend Dependencies

```powershell
cd frontend
npm install
Copy-Item .env.example .env
# Edit .env with VITE_API_URL if needed (default: http://127.0.0.1:5000)
```

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

Frontend runs on `http://localhost:3000` with API proxy to backend.

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

## Backend API

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login (returns JWT token)
- `GET /api/auth/me` - Current user profile (authenticated)

### Assessments
- `POST /api/assessments` - Create assessment (authenticated)
- `GET /api/assessments/<id>` - Get assessment (authenticated)
- `POST /api/assessments/<id>/complete` - Complete assessment (authenticated)
- `POST /api/assessments/<id>/speech` - Update speech step (authenticated)
- `POST /api/assessments/<id>/spiral` - Update spiral step (authenticated)
- `POST /api/assessments/<id>/tremor` - Update tremor step (authenticated)
- `GET /api/assessments/<id>/results` - Get results (authenticated)

### History
- `GET /api/history/assessments` - List user assessments (authenticated)
- `GET /api/history/completed` - List completed assessments (authenticated)
- `GET /api/history/assessments/<id>` - Get specific assessment (authenticated)

### Modality Endpoints
- `GET /api/health` - Health check and model readiness
- `POST /api/speech/predict` - Speech upload and prediction (blocked: no raw-audio extractor)
- `POST /api/spiral/predict` - Spiral/handwriting upload and prediction (blocked: camera/tablet incompatibility)
- `POST /api/tremor/submit` - Manual tremor input (prototype only)

### Legacy Sessions
- `POST /api/assessment/sessions` - Create in-memory session (backward compatibility)
- `GET /api/assessment/sessions/<id>` - Get in-memory session
- `POST /api/assessment/sessions/<id>/complete` - Complete in-memory session
- `GET /api/assessment/sessions/<id>/results` - Get in-memory session results

See [API documentation](app_backend/API.md) and [feature compatibility status](app_backend/FEATURE_COMPATIBILITY.md) for details.

## Frontend Features

- **Authentication**: Registration, login, logout with JWT token management
- **Dashboard**: Assessment status cards for handwriting, speech, and tremor tests
- **Handwriting Test**: Video upload (unavailable due to camera/tablet incompatibility)
- **Speech Test**: Audio upload (unavailable due to missing raw-audio extractor)
- **Tremor Assessment**: Body location and perceived intensity input (self-reported only)
- **Results**: Modality status, fused classification when available, completion workflow
- **History**: Completed assessments with modality status and fused results
- **Design**: Sage-green and cream palette, responsive layout, accessible controls

## Current Status

### Functional
- ML pipeline with reproducible training and evaluation
- Backend authentication with JWT tokens and SQLite persistence
- User-isolated assessment creation, updates, and completion
- Fusion logic for combining available modality predictions
- Frontend authentication and navigation
- Frontend assessment workflow with status tracking

### Blocked/Prototype
- **Speech prediction**: No verified raw-audio feature extractor matching the saved model schema
- **Handwriting prediction**: Camera video cannot provide tablet pressure/grip-angle features
- **Tremor prediction**: Manual input only; no model integration
- **Severity estimation**: Not deployed due to poor regression performance (negative R²)
- **End-to-end assessment**: Cannot complete real assessments due to speech and handwriting blockers

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
- **Tremor**: Manual input only, no model integration
- **Severity**: Poor regression performance (negative R²)
- **Clinical use**: Not validated for clinical diagnosis

## Documentation

- [Architecture](docs/architecture.md) - System architecture diagram
- [Project Overview](PROJECT.md) - Concise architecture and scope
- [Full Report](PROJECT_REPORT.md) - Detailed system documentation
- [API Reference](app_backend/API.md) - Backend API endpoints
- [Feature Compatibility](app_backend/FEATURE_COMPATIBILITY.md) - Model input requirements
- [Frontend README](frontend/README.md) - Frontend setup and usage

## Tests

```powershell
# Backend tests
python -m unittest discover -s tests -v
python -m unittest discover -s app_backend/tests -v

# Frontend (in frontend directory)
npm test  # If configured
npm run build  # Verify build
```

## License

This is a research prototype. See individual dataset licenses for data usage terms.
