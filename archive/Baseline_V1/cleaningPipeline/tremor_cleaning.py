from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "raw_dataset" / "tremors" / "dataset.csv"
OUTPUT = ROOT / "cleaned_dataset" / "tremors_clean.csv"


def clean_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value).replace("\n", " ").lower())
    return value.strip("_") or "unnamed"


def unique_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        result.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return result


def clean_tremors() -> pd.DataFrame:
    raw = pd.read_csv(INPUT, header=None, dtype=str, keep_default_na=False, encoding="latin1")
    groups = raw.iloc[0]
    headers = raw.iloc[1]
    data = raw.iloc[2:].copy()
    names = []
    for group, header in zip(groups, headers):
        header_name = clean_name(header)
        group_name = clean_name(group) if str(group).strip() else ""
        names.append(f"{group_name}_{header_name}" if group_name else header_name)
    data.columns = unique_names(names)
    data = data.apply(lambda column: column.map(lambda value: value.strip() if isinstance(value, str) else value))
    data = data.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "NULL": np.nan, "?": np.nan})
    data = data.dropna(how="all")
    before_duplicates = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    duplicate_count = before_duplicates - len(data)
    for column in data.columns:
        if data[column].dtype == "object":
            numeric = pd.to_numeric(data[column], errors="coerce")
            non_missing = data[column].notna().sum()
            if non_missing and numeric.notna().sum() == non_missing:
                data[column] = numeric
            else:
                data[column] = data[column].replace({"yes": "Yes", "YES": "Yes", "no": "No", "NO": "No"})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    print(f"Tremor cleaning report: raw={raw.shape}, clean={data.shape}, duplicates_removed={duplicate_count}, missing={int(data.isna().sum().sum())}")
    print(f"Saved: {OUTPUT}")
    return data


if __name__ == "__main__":
    clean_tremors()