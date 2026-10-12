from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FOLDS = 5
CALIBRATION_FOLDS = 3
WEIGHTS = [round(i / 10, 1) for i in range(11)]  # SVM share; RF share is 1 - weight
ARTIFACTS = ROOT / "models" / "artifacts"
REPORTS = ROOT / "reports"

CLASSIFICATION_DATASETS = {
    "speech": (ROOT / "cleaned_dataset/speech_clean.csv", "status", "id"),
    "handwriting": (ROOT / "featureEngineering/handwriting_features.csv", "label", "participant_id"),
    "tremor": (ROOT / "cleaned_dataset/tremors_clean.csv", "label", "participant_code"),
}
REGRESSION_DATASET = (ROOT / "cleaned_dataset/telemonitoring_clean.csv", "motor_updrs", "subject_number")
