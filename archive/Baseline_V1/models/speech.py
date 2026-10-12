import pandas as pd

from models.classification import run_classifiers
from models.common import ROOT


def run_speech_models():
    speech = pd.read_csv(ROOT / "cleaned_dataset" / "speech_clean.csv")
    features = speech.drop(columns=["id", "recording", "status"], errors="ignore")
    target = speech["status"].astype(int)
    return run_classifiers("speech", features, target, speech["id"])