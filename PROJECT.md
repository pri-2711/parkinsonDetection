# Parkinson's Detection Project

## Overview

Research prototype for Parkinson's disease classification from speech, handwriting, and tremor, with separate motor-UPDRS severity regression from telemonitoring data. Includes a Flask backend for assessment workflows, user accounts, and a React frontend.

## Architecture

### ML Pipeline (`src/`, `models/`, `cleaningPipeline/`, `featureEngineering/`)
- **Data cleaning**: Deterministic transforms from raw datasets to cleaned tables
- **Feature engineering**: Handwriting trajectory aggregation and tablet feature extraction
- **Model training**: Nested subject-disjoint cross-validation with calibrated SVM and Random Forest
- **Fusion**: Decision-level probability fusion using modality-specific OOF ROC-AUC weights
- **Regression**: Ridge and Random Forest for telemonitoring motor-UPDRS estimation

### Backend (`app_backend/`)
- **Flask API**: REST endpoints for health, auth, assessments, and history
- **Authentication**: Registration, login, JWT tokens, user-isolated data access
- **Persistence**: SQLite database for users and assessments
- **Model loading**: Loads tuned artifacts from `experiments/regularized_svm_rf/results/artifacts/`
- **Validation**: Audio/video upload validation with size, format, and duration limits
- **CORS**: Enabled for frontend integration

### Frontend (`frontend/`)
- **React + TypeScript**: Component-based UI with type safety
- **Authentication**: Registration, login, logout with JWT token management
- **Assessment Workflow**: Dashboard, modality tests, results, and history
- **API Client**: Centralized HTTP client with token handling
- **Styling**: Sage-green and cream palette with responsive design

### Datasets
- **Speech**: 240 recordings, 80 participants, 45 acoustic features
- **Handwriting**: 110 tests, 40 participants, 16 tablet trajectory features
- **Tremor**: 130 participants, 7 UPDRS items (20/21)
- **Telemonitoring**: 5,875 recordings, 42 subjects, 19 voice features for UPDRS regression

## Model Strategy

### Classification
- Five-fold `StratifiedGroupKFold` for subject-disjoint evaluation
- Calibrated RBF SVM and Random Forest with sigmoid calibration
- SVM/RF ensemble weights selected from OOF predictions
- Probability fusion using normalized modality ROC-AUC weights

### Regression
- Five-fold `GroupKFold` for telemonitoring severity estimation
- Ridge and Random Forest compared against group-mean baseline
- Current R² is negative, indicating poor generalization to unseen subjects

## Current Scope

### Functional
- ML pipeline with reproducible training and evaluation
- Fusion configuration from OOF modality performance
- Backend health checks and model loading
- User registration and authentication
- Assessment creation, updates, and completion
- Upload validation for audio and video
- Frontend authentication and navigation
- Frontend assessment workflow with status tracking

### Blocked/Prototype
- **Speech prediction**: Raw audio feature extraction not implemented; model expects precomputed features
- **Handwriting prediction**: Camera video cannot provide tablet pressure/grip-angle features
- **Tremor prediction**: Manual input only; no model integration
- **Severity estimation**: Regression model has negative R², not deployed to backend
- **End-to-end assessment**: Cannot complete due to speech and handwriting blockers

## Modules

- `src/pipeline.py`: Main training entry point
- `src/fusion.py`: Probability fusion logic
- `app_backend/`: Flask API with routes, services, repositories
- `frontend/`: React + TypeScript frontend
- `models/`: Model implementations per modality
- `cleaningPipeline/`: Data cleaning scripts
- `featureEngineering/`: Feature extraction
- `experiments/regularized_svm_rf/`: Tuning experiment with nested validation
- `archive/Baseline_V1/`: Immutable baseline version

## Setup

### Backend

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env with APP_SECRET_KEY and other settings
python -m app_backend
```

### Frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env
# Edit .env with VITE_API_URL if needed (default: http://127.0.0.1:5000)
npm run dev
```

## Documentation

- `README.md`: Installation, usage, and project overview
- `docs/architecture.md`: Detailed system architecture diagram
- `app_backend/API.md`: API endpoint documentation
- `app_backend/FEATURE_COMPATIBILITY.md`: Model compatibility status
- `frontend/README.md`: Frontend setup and usage
- `reports/FINAL_REPORT.md`: Pipeline results and metrics
