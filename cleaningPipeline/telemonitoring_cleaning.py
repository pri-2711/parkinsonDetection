from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "raw_dataset" / "telemonitoring" / "parkinsons_updrs.data"
OUTPUT = ROOT / "cleaned_dataset" / "telemonitoring_clean.csv"


def clean_telemonitoring() -> pd.DataFrame:
    raw = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    data = raw.copy()
    data.columns = [str(column).strip().lower().replace("#", "_number").replace("%", "_percent").replace(":", "_").replace("(", "").replace(")", "").replace(" ", "_") for column in data.columns]
    data = data.apply(lambda column: column.map(lambda value: value.strip() if isinstance(value, str) else value))
    data = data.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "NULL": np.nan, "?": np.nan})
    data = data.dropna(how="all")
    before_duplicates = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    duplicate_count = before_duplicates - len(data)
    for column in data.columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.notna().sum() == data[column].notna().sum():
            data[column] = numeric
    for column in data.select_dtypes(include=np.number).columns:
        if data[column].isna().any():
            data[column] = data[column].fillna(data[column].median())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    print(f"Telemonitoring cleaning report: raw={raw.shape}, clean={data.shape}, subjects={data['subject_number'].nunique()}, duplicates_removed={duplicate_count}, missing={int(data.isna().sum().sum())}")
    print("Subject number preserved for grouped splits and longitudinal analysis.")
    print(f"Saved: {OUTPUT}")
    return data


if __name__ == "__main__":
    clean_telemonitoring()