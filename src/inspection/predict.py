"""Inspect a photo with the same engine used by evaluation and the public demo."""
import argparse
import json
from pathlib import Path
import numpy as np
from .engine import InspectionEngine


def predict(checkpoint_path, image_path, output, device_name="auto"):
    output = Path(output)
    names = ("prediction.json", "heatmap.png", "overlay.png", "anomaly-map.npz")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Prediction artifacts already exist; choose a new output directory")
    inspection = InspectionEngine(checkpoint_path, device_name).inspect(image_path)
    result = inspection["summary"]
    result.update(image=str(Path(image_path).resolve()), checkpoint=str(Path(checkpoint_path).resolve()))
    output.mkdir(parents=True, exist_ok=True)
    inspection["heatmap"].save(output / "heatmap.png")
    inspection["overlay"].save(output / "overlay.png")
    np.savez_compressed(output / "anomaly-map.npz", model=inspection["raw_map"], original=inspection["native_map"])
    (output / "prediction.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
