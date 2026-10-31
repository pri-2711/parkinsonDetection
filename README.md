# Parkinson's Detection and Severity Estimation

Research prototype for Parkinson's disease classification from speech, handwriting, and tremor, with separate motor-UPDRS severity regression from telemonitoring data. Includes a Flask backend with user authentication, assessment workflows, SQLite persistence, and a React frontend.

## Overview

This repository contains:

1. **ML Pipeline**: Reproducible data cleaning, feature engineering, model training, and evaluation with subject-disjoint cross-validation
2. **Flask Backend**: REST API for user accounts, assessments, and model inference
3. **React Frontend**: Web interface for authentication, assessment workflows, and results visualization
4. **Model Artifacts**: Trained classifiers and regression models with documented limitations
5. **Documentation**: Architecture, API reference, and feature compatibility status

**Important**: The classification cohorts are independent and unmatched (no participant appears in more than one modality). No patient-level fusion performance has been measured. Tremor works from clinician-rated UPDRS inputs. Speech has an **experimental** reading-passage model (cross-validated ROC-AUC 0.642 ± 0.144 on 37 participants, close to the majority-class baseline); it is not clinically validated. Handwriting classification needs tablet telemetry, and camera video is observation-only. Telemonitoring severity regression stays disabled.

## How to Open and Run the Project

This section is written for first-time users. Commands are shown for **Windows PowerShell**, which is what the project was developed and tested on. macOS and Linux were not tested; the same steps apply with the platform's own commands.

### 1. Prerequisites

| Software | Version used for testing | Check with |
|---|---|---|
| Python | 3.12 (3.12.2 tested) | `python --version` |
| Node.js and npm | Node.js 18 or newer (Node 24 tested) | `node --version` and `npm --version` |
| Visual Studio Code | any recent version (recommended) | Open the application |
| Git | optional, for version control | `git --version` |

### 2. Open the project in VS Code

1. Start VS Code.
2. Choose **File > Open Folder…** and select the folder that contains this `README.md`, `app_backend`, and `frontend` (the project root, `parkinson_detection_project`).
3. Open a terminal with **Terminal > New Terminal**. It starts in the project root. Use PowerShell.

All commands below are run from the project root unless a step says otherwise.

### 3. Set up the backend (Python)

Create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

### 4. Create the backend configuration (`.env`)

The backend needs a secret key to sign login tokens. Create the local `.env` file once (it is ignored by Git, so keep it private):

```powershell
python -c "import secrets; from pathlib import Path; p=Path('.env'); p.exists() or p.write_text('HOST=127.0.0.1\nPORT=5000\nAPP_SECRET_KEY='+secrets.token_urlsafe(48)+'\nLOG_LEVEL=INFO\n', encoding='utf-8')"
```

If `.env` already exists, the command leaves it unchanged. `.env.example` lists every optional setting. The most useful ones:

| Variable | Default | Purpose |
|---|---|---|
| `APP_SECRET_KEY` | required | Signs login tokens. Login fails without it. |
| `HOST`, `PORT` | `127.0.0.1`, `5000` | Backend address. |
| `DATABASE_PATH` | `data/parkinson_app.db` | SQLite database file. |
| `CORS_ORIGINS` | unset: `localhost` and `127.0.0.1` on any port | Browser origins allowed to call the API. Set a comma-separated list only for known deployment origins. |
| `MODEL_ARTIFACT_DIR` | `experiments/regularized_svm_rf/results/artifacts` | Tremor, handwriting and legacy classifiers. |
| `SPEECH_AUDIO_V2_ARTIFACT_DIR` | `experiments/speech_audio_v2/artifacts` | Speech reading-model files. |
| `HAND_LANDMARK_PYTHON` | `.venv-mediapipe` | Optional. Python executable with MediaPipe installed, used only for spiral-video hand landmarks. Defaults to the repository `.venv-mediapipe` when it exists. |

### 5. Database

The database is created automatically the first time the backend starts: the `data` folder and the tables are created if missing. Existing data is never reset. Your accounts and assessments live in `data/parkinson_app.db`, which Git ignores. To keep a backup, stop the backend and copy that file. Do not delete it if you want to keep your accounts.

### 6. Model files

- Tuned classifiers (tremor, handwriting, legacy speech baseline): `experiments/regularized_svm_rf/results/artifacts/`.
- Speech reading-passage model: `experiments/speech_audio_v2/artifacts/reading.joblib` with its `validation_metrics.json`. The application loads it at startup and shows it as experimental. Moving `reading.joblib` out of that folder turns speech prediction off.
- Do not run the training scripts unless you intend to overwrite these files. `experiments/regularized_svm_rf/run_experiments.py` regenerates the tuned results.

### 7. Install and start the frontend (second terminal)

Open a **second** terminal in VS Code (**Terminal > New Terminal**) and run:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

`npm ci` installs the exact versions in `package-lock.json`. Leave this terminal running. `frontend/.env` sets `VITE_API_URL=http://127.0.0.1:5000`, the address of the backend.

### 8. Start the backend (first terminal)

Return to the **first** terminal, which has the virtual environment active, and run:

```powershell
python -m app_backend
```

Leave it running. If the backend started, `http://127.0.0.1:5000/api/health` returns a JSON status in your browser.

### 9. Open the application

| What | Address |
|---|---|
| Web application | `http://localhost:3000` (use `localhost`; the Vite dev server listens on it) |
| Backend API root | `http://127.0.0.1:5000/api` |
| Health check | `http://127.0.0.1:5000/api/health` |

1. Open `http://localhost:3000` and select **Register**. Passwords need at least 8 characters.
2. Log in. The Dashboard shows tremor, handwriting/spiral and speech steps.
3. Complete any steps you want, then select **View Assessment Results & Status**. From Results you can finalize the assessment and download the PDF report. History lists your assessments.
4. Research & Insights shows the saved experiment metrics and the speech model's validation limits.

Login sessions last 24 hours and cannot be renewed past 7 days from login. The browser keeps the session only for the current tab, so you will need to log in again after closing it.

### 10. Stop the servers

Press `Ctrl+C` in each terminal.

### 11. Run the tests

From the project root, with the virtual environment active for the Python commands:

```powershell
python -m unittest discover -s app_backend/tests -t .
python -m unittest app_backend.tests.test_speech_and_reporting -v
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/regularized_svm_rf -p "test_*.py" -v
python -m pytest -q experiments/regularized_svm_rf/test_experiments.py tests/test_fusion.py
```

Frontend, from the `frontend` folder:

```powershell
node --test src/features/*.test.mjs
npm run build
```

The backend suite takes about 2 minutes because it trains a small synthetic speech model. `npm run lint` does not work: the repository has no ESLint configuration.

### 12. Regenerate the Word project report

From the project root (the virtual environment must have the requirements installed):

```powershell
python scripts/build_project_report.py
```

This writes the editable report to `docs/Project_Report.docx`. It reads routes, metrics and dataset sizes from the repository, so it reflects the current code.

### Troubleshooting

- **`Activate.ps1 cannot be loaded`**: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
- **Login fails with a server-configuration error**: `.env` is missing `APP_SECRET_KEY`. Repeat step 4.
- **Frontend shows "Could not reach the backend"**: the backend is not running, or `frontend/.env` points at another address.
- **Port already in use**: another program uses port 5000 or 3000. Stop it, or set `PORT` in `.env` and `VITE_API_URL` in `frontend/.env` to match.

## Installation

The step-by-step setup is in **How to Open and Run the Project** above. The short form:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm ci
```

### Backend Dependencies

```powershell
python -m pip install -r requirements.txt
python -c "import secrets; from pathlib import Path; p=Path('.env'); p.exists() or p.write_text('HOST=127.0.0.1\nPORT=5000\nAPP_SECRET_KEY='+secrets.token_urlsafe(48)+'\nLOG_LEVEL=INFO\n', encoding='utf-8')"
```

The backend reads `APP_SECRET_KEY` from the repository-root `.env` when the Flask app is created. The generated `.env` is ignored by Git; do not share it. Existing accounts and password hashes stay in `data/parkinson_app.db`.

`VITE_API_URL` defaults to `http://127.0.0.1:5000`; only change it when the backend runs elsewhere.

#### Optional: hand landmarks for spiral videos

Hand landmarks use MediaPipe Hands. MediaPipe pins its own NumPy and OpenCV versions, which conflict with the backend, so it runs in a separate environment. Without it, the video analysis still measures ink and the spiral, and reports hand values as not measured.

```powershell
python -m venv .venv-mediapipe
.\.venv-mediapipe\Scripts\python.exe -m pip install mediapipe==0.10.21 numpy==1.26.4
```

The server uses `.venv-mediapipe` in the repository root automatically. To use another interpreter, set `HAND_LANDMARK_PYTHON` in `.env`. The hand model ships inside the MediaPipe wheel (`hand_landmark_full.tflite`, Apache-2.0), so no separate download is needed. `GET /api/health` reports whether an interpreter was found. If the worker cannot run or fails, the analysis shows the reason under Hand & Grip Analysis instead of a detection rate.

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

Frontend runs on `http://localhost:3000`; the Vite proxy forwards `/api` requests to Flask on port 5000.

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
- **Speech prediction (experimental)**: Praat/SciPy extracts the 45-feature `speech_audio_v2` schema from WAV input (44 model inputs; gender is not used by the model). A reading-passage model trained on the public MDVR-KCL recordings (37 participants) is loaded. Its subject-disjoint ROC-AUC is 0.642 ± 0.144, close to the majority-class baseline of 0.605. Sustained-vowel prediction is unavailable. Recordings are not persisted. See `experiments/speech_audio_v2/README.md`.
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

- **Small cohorts**: 40 handwriting participants, 37 speech participants (reading model), 130 tremor participants
- **Unmatched cohorts**: No patient-level fusion performance measured
- **Speech**: The reading-passage model is weak and unstable across folds, was trained on an external public cohort, and has not been tested on app recordings. Sustained-vowel data is missing.
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

The commands are listed in step 11 of **How to Open and Run the Project**. The frontend production build (`npm run build`) succeeds. The lint command (`npm run lint`) cannot run because the repository has no ESLint configuration.

## License

This is a research prototype. See individual dataset licenses for data usage terms.
