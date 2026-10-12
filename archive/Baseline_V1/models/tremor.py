import pandas as pd

from models.classification import run_classifiers
from models.common import ROOT


def run_tremor_models():
    tremor = pd.read_csv(ROOT / "cleaned_dataset" / "tremors_clean.csv")
    tremor["label"] = tremor["participant_code"].astype(str).str.startswith("PD").astype(int)
    tremor_features = [
        column for column in tremor.columns
        if "motor_scale_specific_items_20_" in column or "motor_scale_specific_items_21_" in column
    ]
    if not tremor_features:
        raise ValueError("No tremor-specific UPDRS items (20/21) were found in the tremor dataset")
    features = tremor[tremor_features]
    target = tremor["label"]
    return run_classifiers("tremor", features, target, tremor["participant_code"])