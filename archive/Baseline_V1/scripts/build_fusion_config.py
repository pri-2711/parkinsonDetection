from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion import build_fusion_artifacts


if __name__ == "__main__":
    config = build_fusion_artifacts()
    print("Fusion weights:", config["normalized_weights"])
