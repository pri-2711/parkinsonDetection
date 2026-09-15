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

## Implementation

1. Clean the three current datasets.
2. Engineer handwriting features.
3. Train and evaluate the three PD classifiers.
4. Train the fusion model using classifier probabilities.
5. Process the Telemonitoring dataset and train the UPDRS regression model.
6. Integrate the models into the application.
7. Evaluate the complete system.

Subject-level splitting will be used where multiple recordings belong to the same subject to prevent data leakage.

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
