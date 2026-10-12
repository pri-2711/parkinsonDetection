import pandas as pd

from models.classification import run_classifiers
from models.common import ROOT


def run_handwriting_models():
    handwriting = pd.read_csv(ROOT / "featureEngineering" / "handwriting_features.csv")
    features = handwriting.drop(columns=["participant_id", "test_id", "label"], errors="ignore")
    target = handwriting["label"].astype(int)
    return run_classifiers("handwriting", features, target, handwriting["participant_id"])