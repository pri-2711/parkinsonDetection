"""Prepare labelled reading-passage segments from the MDVR-KCL archive for speech_audio_v2 training.

Source: "Mobile Device Voice Recordings at King's College London (MDVR-KCL)", Zenodo record 2867216,
licence CC-BY-4.0. The ReadText folder holds one passage-reading recording per participant
(37 participants: 21 healthy controls, 16 Parkinson's). Each recording is split into
non-overlapping windows of --segment-seconds (default 20 s, inside the app's 30 s upload limit).

Only ReadText is used. SpontaneousDialogue is not a fixed reading task and is excluded.
The archive has no sex or age metadata, so gender_code is written as a placeholder and the
trained bundle does not use gender (see experiments/speech_audio_v2/README.md).

usage: python experiments/speech_audio_v2/prepare_mdvr.py <mdvr.zip> <output_dir> [--segment-seconds 20]
"""
from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import soundfile as sf

FILENAME = re.compile(r"(?P<subject>ID\d+)_(?P<label>hc|pd)_", re.IGNORECASE)


def prepare(archive: Path, output: Path, segment_seconds: float = 20.0, minimum_seconds: float = 10.0) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive) as zf:
        members = sorted(n for n in zf.namelist() if "/ReadText/" in n and n.lower().endswith(".wav"))
        if not members:
            raise ValueError("No ReadText WAV recordings found in the archive")
        for member in members:
            match = FILENAME.search(Path(member).name)
            folder_label = "pd" if "/PD/" in member else "hc" if "/HC/" in member else None
            if not match or folder_label is None or match.group("label").lower() != folder_label:
                raise ValueError(f"Cannot determine a consistent label for {member}")
            subject = match.group("subject").upper()
            status = 1 if folder_label == "pd" else 0
            audio, rate = sf.read(io.BytesIO(zf.read(member)), dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            window = int(segment_seconds * rate)
            for index, start in enumerate(range(0, len(mono), window)):
                chunk = mono[start:start + window]
                if len(chunk) < minimum_seconds * rate:
                    continue
                name = f"{subject}_{index:02d}.wav"
                sf.write(output / name, chunk, rate, subtype="PCM_16")
                rows.append({"audio_path": name, "subject_id": f"MDVR-{subject}", "mode": "reading",
                             "gender_code": 0, "status": status})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output / "manifest.csv", index=False)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--segment-seconds", type=float, default=20.0)
    args = parser.parse_args()
    table = prepare(args.archive, args.output, args.segment_seconds)
    print(f"{len(table)} segments from {table['subject_id'].nunique()} subjects; "
          f"PD subjects {table.drop_duplicates('subject_id')['status'].sum()}")
