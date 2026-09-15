# Parkinson's Detection and Severity Estimation

## Overview
A multimodal machine learning system for Parkinson's Disease (PD) detection using **tremor, speech, and handwriting**. Each modality has an individual classifier, and their outputs are combined using a fusion model.

A separate regression model estimates **Motor UPDRS** using the Parkinson's Telemonitoring dataset.

## Datasets

### Tremor
- ~130 participants
- 65 variables
- Used for PD vs Healthy classification

### Speech
- 240 recordings from 80 subjects
- ~48 acoustic feature columns
- Includes MFCC, jitter, shimmer, HNR, RPDE, DFA and other features
- `Status`: 0 = Healthy, 1 = Parkinson's
- Used for PD classification

### Handwriting
- 40 participants: 15 Control and 25 Parkinson's
- X, Y, Z, Pressure, GripAngle, Timestamp and Test ID
- Requires feature engineering before classification

### Severity Dataset
**Parkinson's Telemonitoring Dataset**
- ~5,875 voice recordings from 42 subjects
- Contains voice features and `motor_UPDRS` / `total_UPDRS`
- Used for severity regression
- Initial target: **Motor UPDRS**

## Architecture

```text
Handwriting → Classifier ─┐
Speech     → Classifier ──┼→ Fusion Model → PD Probability
Tremor     → Classifier ──┘

Voice → UPDRS Regression Model → Estimated Motor UPDRS
```

The classification and severity tracks are trained independently.

## User Testing

1. **Handwriting:** User performs a spiral drawing; movement and pressure data are converted into engineered features.
2. **Speech:** User provides a voice recording; acoustic features are extracted automatically.
3. **Tremor:** Manual tremor input in the current version. Automatic sensor-based measurement is future scope.

## Preprocessing Pipeline

The repository separates raw data, cleaning, feature engineering, and later model preprocessing:

```text
raw_dataset/ -> cleaningPipeline/ -> cleaned_dataset/ -> featureEngineering/ -> model training
```

Run the cleaners from the project root:

```text
python cleaningPipeline/tremor_cleaning.py
python cleaningPipeline/speech_cleaning.py
python cleaningPipeline/handwriting_cleaning.py
python cleaningPipeline/telemonitoring_cleaning.py
python featureEngineering/handwriting_features.py
```

The cleaning scripts do not fit scalers or train models. `cleaningPipeline/preprocessing.py` provides reusable scikit-learn preprocessing helpers. Call `fit_on_training_data` only with the training partition, use `transform` for validation/test data, and use `save_preprocessor` for deployment artifacts.

Handwriting cleaning uses `raw_dataset/handwriting+spiralTest/hw_dataset`, which contains the stated 15 control and 25 Parkinson participant files. The cleaned trajectory table intentionally retains ordered points and participant/test keys. `featureEngineering/handwriting_features.csv` is the participant/test-level table used for later modeling. The Parkinson-only `new_dataset` folder is left untouched and is not mixed into the balanced source used here.

Speech cleaning preserves `id` and `recording`. The 240 recordings from 80 subjects must be split by `id`, never randomly by row. The same subject-level grouping must be used for cross-validation.

Telemonitoring cleaning preserves `subject_number` so longitudinal recordings can also be grouped by subject during later model evaluation.

## Evaluation

**Classification:** Accuracy, Precision, Recall, F1-score, ROC-AUC and Confusion Matrix.

**Regression:** MAE, RMSE and R².

## Technology Stack

- Python
- NumPy, Pandas
- Scikit-learn
- SciPy
- Imbalanced-learn
- Praat-Parselmouth / openSMILE / Librosa
- Matplotlib, Seaborn
- Flask or FastAPI
- Joblib

## Future Scope

- IoT-based tremor measurement using accelerometer/gyroscope sensors
- Larger clinically labelled datasets
- Improved multimodal severity estimation
- Mobile/cloud deployment

> This system is intended for research and screening purposes and is not a substitute for clinical diagnosis.
