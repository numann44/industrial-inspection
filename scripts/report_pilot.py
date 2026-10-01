"""Create a portable, honest first-pilot report from measured run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--all-audit", type=Path, required=True)
    parser.add_argument("--download-report", type=Path, default=Path("data/download-verification.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/results"))
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    history = json.loads((args.run / "history.json").read_text())
    evaluation = json.loads((args.run / "evaluation.json").read_text())
    audit = json.loads(args.all_audit.read_text())
    download = json.loads(args.download_report.read_text())
    code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for split in ("train", "validation"):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], label=split)
    axes[0].set(title="Total loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    for component in ("reconstruction", "segmentation_bce", "segmentation_dice"):
        axes[1].plot(epochs, [row["validation"][component] for row in history], label=component)
    axes[1].set(title="Fixed synthetic validation components", xlabel="Epoch", ylabel="Loss")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(args.output / "pilot-learning-curves.png", dpi=160)
    plt.close(fig)

    groups = evaluation["recall_by_defect"]
    names = [key.split("/")[-1] for key in groups]
    values = [group["recall"] for group in groups.values()]
    fig, axis = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    bars = axis.bar(names, values, color="#2467a6")
    axis.set(title="Exploratory real-defect recall at the frozen threshold", ylabel="Recall", ylim=(0, 1.15))
    for bar, group in zip(bars, groups.values()):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                  f"{group['detected']}/{group['images']}", ha="center")
    fig.savefig(args.output / "pilot-defect-recall.png", dpi=160)
    plt.close(fig)

    counts = audit["counts"]
    normal_training = sum(counts[name]["total"] for name in ("train", "validation", "calibration"))
    test_counts = counts["test"]["by_category_defect"]
    normal_test = sum(count for key, count in test_counts.items() if key.endswith("/good"))
    defect_test = counts["test"]["total"] - normal_test
    expected = (3629, 467, 1258)
    observed = (normal_training, normal_test, defect_test)
    if observed != expected:
        raise ValueError(f"Complete dataset counts differ from paper: {observed} vs {expected}")
    data_summary = {
        "status": audit["status"], "archive_sha256": download["sha256"],
        "compressed_size_bytes": download["size_bytes"],
        "archive_verified": download["archive_verified"], "categories": download["categories"],
        "original_counts": {"normal_training": normal_training, "normal_test": normal_test,
                            "defective_test": defect_test, "total_images": sum(observed)},
        "decoded_pngs_including_masks": audit["decoded_pngs"],
        "counts": counts, "checks": audit["checks"],
        "cross_split_duplicate_hashes": audit["cross_split_duplicate_hashes"],
        "license": download["license"],
    }
    for key in ("masks_vanishing_after_resize", "image_sizes"):
        if key in audit:
            data_summary[key] = audit[key]
    (args.output / "data-verification.json").write_text(json.dumps(data_summary, indent=2) + "\n")

    portable_metrics = {key: value for key, value in evaluation.items()
                        if key not in {"predictions", "checkpoint", "manifest_digest"}}
    portable_metrics.update({"status": "exploratory initial pilot", "code_revision": code_revision,
                             "checkpoint_sha256": hashlib.sha256((args.run / "checkpoint.pt").read_bytes()).hexdigest()})
    (args.output / "pilot-metrics.json").write_text(json.dumps(portable_metrics, indent=2) + "\n")

    confusion = evaluation["confusion"]
    calibration = summary["calibration"]
    best = history[summary["best_epoch"] - 1]
    rows = "\n".join(f"| {name.split('/')[-1]} | {group['detected']} / {group['images']} | {group['recall']:.3f} |"
                     for name, group in groups.items())
    text = f"""# First training pilot

This is an **exploratory initial run**, not a final benchmark or production model. It verifies the complete data-to-training-to-inspection path and identifies weaknesses for the predeclared experiments.

## Data is available locally

The complete MVTec AD archive matches the published SHA-256. All 15 categories were extracted and audited: **{sum(observed):,} images** ({normal_training:,} original normal training, {normal_test:,} normal test and {defect_test:,} defective test), plus the associated defect masks. **{audit['decoded_pngs']:,} PNG files** decoded successfully, including masks. Original counts agree with the official paper. No identical image bytes cross our normal split or test boundaries.

Full portable evidence: [data-verification.json](data-verification.json). Raw data remains local under its CC BY-NC-SA 4.0 license.

## Run

- Category: `{summary['config']['category']}`; input resolution: {summary['image_size']} × {summary['image_size']}.
- Random initialization; compact reconstruction and segmentation networks; no pretrained weights or external texture data.
- Training epochs: {len(history)}; best epoch selected by fixed synthetic-validation loss: {summary['best_epoch']}.
- Device: `{summary['device']}`; PyTorch: `{summary['torch_version']}`.
- Measured training and calibration duration: {summary['elapsed_seconds']:.1f} seconds, excluding download and dataset preparation.
- Validation loss: {history[0]['validation']['loss']:.4f} after epoch 1 → {best['validation']['loss']:.4f} at the selected epoch.
- Code revision: `{code_revision}`.
- Portable split digest: `{summary['split_digest']}`.

![Learning curves](pilot-learning-curves.png)

## Frozen operating threshold

The threshold was fitted exclusively on {calibration['count']} held-out normal calibration images using the {calibration['quantile']:.0%} empirical quantile, `{calibration['quantile_method']}` interpolation. Threshold: **{evaluation['threshold']:.6f}**. Decision rule: score ≥ threshold. Calibration exceedances: {calibration['calibration_exceedances']}/{calibration['count']}. This small set does not guarantee a 5% future false-alarm rate. Scores and map activations are not calibrated probabilities.

## Exploratory real-defect test

| Measurement | Result |
| --- | ---: |
| Image AUROC | {evaluation['image_auroc']:.4f} |
| Original-resolution pixel average precision | {evaluation['pixel_average_precision']:.4f} |
| Defective parts detected | {confusion['true_positive']} |
| Defective parts missed | {confusion['false_negative']} |
| Good parts falsely flagged | {confusion['false_positive']} |
| Good parts accepted | {confusion['true_negative']} |

| Defect group | Detected / available | Recall |
| --- | ---: | ---: |
{rows}

![Recall by defect group](pilot-defect-recall.png)

Metrics and timing scope: [pilot-metrics.json](pilot-metrics.json). Pixel metrics preserve original masks and upsample predictions; upsampling cannot restore tiny defects already lost from the resized input.

## Next work

Compare the declared synthesis variants, investigate part/background corruption coverage, add the reconstruction and pretrained-feature baselines, and measure resolution and seed sensitivity. Keep original test observations explicitly exploratory during development. The screw and transistor categories remain available for later experiments and an additional untouched-category generalization check.

Do not treat declining synthetic-validation loss as proof of real-defect performance. The confusion counts and defect-group results above determine which failure cases need attention. The project has not yet established robustness to new cameras, lighting, arbitrary part types or factory operation.
"""
    (args.output / "FIRST_PILOT.md").write_text(text)
    print(json.dumps({"report": str(args.output / "FIRST_PILOT.md"), "data_counts_verified": observed}, indent=2))


if __name__ == "__main__":
    main()
