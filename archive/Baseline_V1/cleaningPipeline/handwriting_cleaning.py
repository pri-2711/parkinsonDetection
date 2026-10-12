from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "raw_dataset" / "handwriting+spiralTest" / "hw_dataset"
OUTPUT = ROOT / "cleaned_dataset" / "handwriting_clean.csv"
TRAJECTORY_COLUMNS = ["x", "y", "z", "pressure", "grip_angle", "timestamp", "test_id"]


def participant_from_name(path: Path) -> str:
    return re.sub(r"\.(txt|csv)$", "", path.name, flags=re.IGNORECASE)


def read_file(path: Path, label: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", header=None, names=TRAJECTORY_COLUMNS, dtype=str, keep_default_na=False)
    frame = frame.apply(lambda column: column.map(lambda value: value.strip() if isinstance(value, str) else value))
    frame = frame.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "NULL": np.nan, "?": np.nan})
    frame = frame.dropna(how="all")
    frame["participant_id"] = participant_from_name(path)
    frame["label"] = label
    frame["point_index"] = np.arange(len(frame), dtype=np.int64)
    for column in TRAJECTORY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["x", "y", "timestamp", "test_id"])
    frame["test_id"] = frame["test_id"].astype("int16")
    frame["label"] = frame["label"].astype("int8")
    return frame


def clean_handwriting() -> pd.DataFrame:
    frames = []
    for label, folder in ((0, "control"), (1, "parkinson")):
        for path in sorted((INPUT / folder).glob("*.txt")):
            frame = read_file(path, label)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise RuntimeError(f"No handwriting trajectory files found under {INPUT}")
    data = pd.concat(frames, ignore_index=True)
    before_duplicates = len(data)
    data = data.drop_duplicates(subset=["participant_id", "test_id", "point_index"]).reset_index(drop=True)
    duplicate_count = before_duplicates - len(data)
    data = data[["participant_id", "label", "test_id", "point_index", *TRAJECTORY_COLUMNS]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    groups = data[["participant_id", "test_id", "label"]].drop_duplicates()
    print(f"Handwriting cleaning report: files={len(frames)}, trajectory_rows={len(data)}, participant_tests={len(groups)}, duplicates_removed={duplicate_count}, missing={int(data.isna().sum().sum())}")
    print("Participant/test boundaries preserved; X/Y/Z points remain ordered trajectory rows.")
    print(f"Saved: {OUTPUT}")
    return data


if __name__ == "__main__":
    clean_handwriting()