# System Architecture

## Overview

The Parkinson's Detection and Severity Estimation system consists of:

1. **ML Pipeline**: Data cleaning, feature engineering, model training, and evaluation
2. **Flask Backend**: REST API for assessments, authentication, and results
3. **Database**: SQLite persistence for users and assessments
4. **Model Artifacts**: Trained classifiers and regression models

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

## Backend Components

### Routes
- `health.py`: Health check and model readiness
- `auth.py`: User registration, login, and profile
- `speech.py`: Speech upload and prediction (currently blocked by missing feature extractor)
- `spiral.py`: Spiral/handwriting upload and prediction (currently blocked by camera-to-tablet incompatibility)
- `tremor.py`: Manual tremor input (prototype only)
- `assessment.py`: Assessment creation, completion, and state management
- `history.py`: User assessment history retrieval
- `results.py`: Assessment results retrieval

### Services
- `auth.py`: Authentication, password hashing, JWT token management
- `classification_service.py`: Modality classification predictions
- `fusion_service.py`: Fusion of modality probabilities
- `severity_service.py`: UPDRS severity estimation
- `history_service.py`: Assessment history queries
- `sessions.py`: In-memory session management (legacy, retained for compatibility)
- `audio_preprocessing.py`: WAV validation
- `acoustic_features.py`: Placeholder for acoustic feature extraction
- `spiral_preprocessing.py`: Image/video validation
- `trajectory_features.py`: Tablet trajectory feature extraction
- `tremor_placeholder.py`: Tremor input validation

### Repositories
- `user_repository.py`: User data access layer
- `assessment_repository.py`: Assessment data access layer

### Database
- SQLite (`data/parkinson_app.db`)
- Tables: `users`, `assessments`
- User-isolated data access

## ML Pipeline Components

### Data Processing
- `cleaningPipeline/`: Raw data cleaning scripts
- `featureEngineering/`: Feature extraction and aggregation

### Models
- `src/pipeline.py`: Main training pipeline with nested CV
- `src/fusion.py`: Decision-level fusion logic
- `models/`: Model implementations for each modality

### Artifacts
- `models/artifacts/`: Trained model bundles and OOF predictions
- `experiments/regularized_svm_rf/results/artifacts/`: Tuned model artifacts

## Data Flow

1. User uploads data (audio, video, or manual input)
2. Backend validates uploads (type, size, format)
3. If compatible, features are extracted and passed to trained models
4. Modality-specific predictions are generated
5. Available predictions are fused using validated weights
6. Results are stored in the database per user
7. Users can retrieve history and trends

## Known Limitations

- **Speech**: Raw audio feature extraction not yet implemented; current model expects precomputed acoustic features
- **Handwriting**: Camera video cannot provide tablet-specific features (pressure, grip angle); prediction blocked
- **Tremor**: Manual input is a prototype; no trained model integration
- **Severity**: Telemonitoring regression has negative R², indicating poor generalization to unseen subjects
- **Fusion**: Cohorts are unmatched; no patient-level fusion performance measured
- **Authentication**: Basic JWT implementation; no refresh tokens or multi-factor auth
