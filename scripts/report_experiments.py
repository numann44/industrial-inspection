"""Build portable comparison evidence and balanced galleries from frozen runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from inspection.engine import InspectionEngine
from inspection.evaluate import load_evaluation_manifest
from inspection.reporting import defect_area_groups, image_uncertainty, write_failure_gallery


def _format(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(run_directories, manifest_path, output, *, device="cpu", galleries=True, bootstrap_samples=1000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_evaluation_manifest(manifest_path, verify_splits=("test",))
    records_by_path = {record["image"]: record for record in manifest["splits"]["test"]}
    root = Path(manifest["root"])
    reports = []
    for directory in map(Path, run_directories):
        evaluation = json.loads((directory / "evaluation.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        history_path = directory / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        predictions = evaluation["predictions"]
        records = [records_by_path[prediction["image"]] for prediction in predictions]
        if len(records) != len(manifest["splits"]["test"]) or len({record["image"] for record in records}) != len(records):
            raise ValueError("Each report requires every original test record exactly once")
        if evaluation["split_digest"] != manifest["split_digest"]:
            raise ValueError("Compared runs must use the same audited test split")
        labels = [prediction["label"] for prediction in predictions]
        scores = [prediction["score"] for prediction in predictions]
        if labels != [record["label"] for record in records]:
            raise ValueError("Prediction labels do not match original audited test records")
        threshold = evaluation["threshold"]
        checkpoint_path = directory / "checkpoint.pt"
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if evaluation.get("checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
            raise ValueError("Evaluation checkpoint checksum differs from the current checkpoint")
        uncertainty = evaluation.get("uncertainty") or image_uncertainty(labels, scores, threshold, bootstrap_samples=bootstrap_samples)
        areas = evaluation.get("defect_area_groups") or defect_area_groups(records, scores, threshold)
        portable_metrics = {key: value for key, value in evaluation.items()
                            if key not in {"predictions", "checkpoint", "manifest_digest", "gallery"}}
        portable_metrics.update({"uncertainty": uncertainty, "defect_area_groups": areas,
                                 "checkpoint_sha256": checkpoint_hash,
                                 "status": "exploratory metal_nut development" if records[0]["category"] == "metal_nut"
                                 else evaluation.get("experimental_status", "evaluation status unspecified")})
        portable_predictions = [{**prediction, "image": Path(prediction["image"]).relative_to(root).as_posix()}
                                for prediction in predictions]
        config_path = directory / "config.json"
        config = summary.get("config") or (json.loads(config_path.read_text()) if config_path.exists() else {})
        report = {"run": directory.name, "config": config, "model_kind": summary.get("model_kind", "joint_reconstruction_segmentation"),
                  "image_size": summary.get("image_size", config.get("image_size")),
                  "epochs": len(history), "best_epoch": summary.get("best_epoch"),
                  "model_selection": summary.get("model_selection", "minimum per-method normal/synthetic validation loss"),
                  "bank_digest": summary.get("bank_digest"), "best_metric": summary.get("best_metric"),
                  "calibration_quantile": summary.get("calibration", {}).get("quantile", config.get("threshold_quantile")),
                  "calibration_count": summary.get("calibration", {}).get("count"),
                  "training_seconds": summary.get("elapsed_seconds", summary.get("seconds")), "metrics": portable_metrics,
                  "predictions": portable_predictions}
        if galleries:
            engine = InspectionEngine(checkpoint_path, device=device)
            maps = [None] * len(records)
            from inspection.reporting import select_gallery_cases
            groups = select_gallery_cases(records, scores, threshold, limit_per_group=2)
            for index in sorted({index for indices in groups.values() for index in indices}):
                maps[index] = engine.inspect(records[index]["image"])["native_map"]
            gallery = write_failure_gallery(records, maps, scores, threshold, output / "galleries" / directory.name,
                                            limit_per_group=2, display_max=engine.display_max)
            report["gallery"] = gallery
            report["gallery_map_generation"] = f"Same checkpoint recomputed with shared InspectionEngine on {device}; decisions/scores come from frozen evaluation JSON"
        else:
            index_path = output / "galleries" / directory.name / "index.json"
            if index_path.exists():
                cached_gallery = json.loads(index_path.read_text())
                if cached_gallery["threshold"] != threshold:
                    raise ValueError("Cached gallery threshold differs; regenerate galleries")
                report["gallery"] = cached_gallery
                report["gallery_map_generation"] = "Reused existing gallery with matching frozen threshold; no new inference in this report invocation"
        report["history"] = history
        reports.append(report)

    fig, axes = plt.subplots(1, len(reports), figsize=(max(6, len(reports) * 4), 4), squeeze=False, constrained_layout=True)
    for axis, report in zip(axes[0], reports):
        history = report["history"]
        if history:
            for split in ("train", "validation"):
                if f"{split}_l1" in history[0]:
                    values = [row[f"{split}_l1"] for row in history]
                else:
                    values = [row[split]["loss"] for row in history]
                axis.plot([row["epoch"] for row in history], values, label=split)
            axis.legend()
        else:
            axis.text(0.5, 0.5, "Non-parametric fitting\nNo epoch loss curve", ha="center", va="center")
        axis.set(title=report["run"], xlabel="Epoch", ylabel="Per-method objective (not comparable across methods)")
        axis.grid(alpha=0.2)
    fig.savefig(output / "experiment-learning-curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for report in reports:
        labels = [prediction["label"] for prediction in report["predictions"]]
        scores = [prediction["score"] for prediction in report["predictions"]]
        false_alarm, recall, _ = roc_curve(labels, scores)
        axes[0].plot(false_alarm, recall, label=f"{report['run']} ({report['metrics']['image_auroc']:.3f})")
        axes[1].scatter(report["metrics"]["normal_false_alarm_rate"], report["metrics"]["defect_recall"], label=report["run"])
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(title="Exploratory image ranking", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(title="Frozen operating thresholds", xlabel="Normal false-alarm rate", ylabel="Defect recall", xlim=(0, 1), ylim=(0, 1))
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    fig.savefig(output / "experiment-ranking-and-decisions.png", dpi=160)
    plt.close(fig)

    groups = sorted({name for report in reports for name in report["metrics"]["recall_by_defect"]})
    fig, axis = plt.subplots(figsize=(max(8, len(groups) * 1.5), 4.5), constrained_layout=True)
    positions = np.arange(len(groups))
    width = 0.8 / len(reports)
    for index, report in enumerate(reports):
        values = [report["metrics"]["recall_by_defect"].get(name, {}).get("recall", 0) for name in groups]
        axis.bar(positions + (index - (len(reports) - 1) / 2) * width, values, width, label=report["run"])
    axis.set(xticks=positions, xticklabels=[name.split("/")[-1] for name in groups], ylabel="Defect recall", ylim=(0, 1.12),
             title="Recall by annotated defect group at frozen thresholds")
    axis.legend(fontsize=7)
    fig.savefig(output / "experiment-defect-recall.png", dpi=160)
    plt.close(fig)

    rows, operating_rows = [], []
    for report in reports:
        metrics = report["metrics"]
        interval = metrics["uncertainty"]["image_auroc_interval"]
        ci = f"[{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "n/a"
        rows.append(f"| {report['run']} | {report['image_size']} | {report['epochs']} / {report['best_epoch']} | "
                    f"{metrics['image_auroc']:.3f} {ci} | {_format(metrics['pixel_average_precision'])} |")
        confusion = metrics["confusion"]
        operating_rows.append(f"| {report['run']} | {report['calibration_quantile']} | {confusion['true_positive']} / "
                              f"{confusion['true_positive'] + confusion['false_negative']} | {confusion['false_positive']} / "
                              f"{confusion['false_positive'] + confusion['true_negative']} | {_format(metrics['precision'])} |")
    prevalence = float(np.mean([record["label"] for record in manifest["splits"]["test"]]))
    text = "# Frozen experiment evidence\n\n"
    text += "These metal-nut results are **exploratory development measurements**. The category's test set has been inspected during development; these are not blind final benchmark results. All weights and thresholds were frozen before each reported evaluation.\n\n"
    text += "| Run | Input size | Trained / selected epoch | Image AUROC [95% image bootstrap interval] | Native-mask pixel AP |\n| --- | --- | --- | --- | --- |\n" + "\n".join(rows) + "\n\n"
    text += "| Run | Normal calibration quantile | Defects detected / available | Good parts falsely flagged / available | Precision |\n| --- | --- | --- | --- | --- |\n" + "\n".join(operating_rows) + "\n\n"
    text += f"Defective-image prevalence is {prevalence:.1%}; image average precision and positive predictive value depend on this unusually defect-heavy test mix. An uninformative image ranking has an AP reference approximately equal to this prevalence. The normal calibration sample size is small, so empirical quantiles do not guarantee future false-alarm rates. Full rate intervals, native-mask area groups, checkpoint checksums and portable prediction records are in [experiment-metrics.json](experiment-metrics.json).\n\n"
    text += "![Per-method learning objectives](experiment-learning-curves.png)\n\n![Ranking and frozen decisions](experiment-ranking-and-decisions.png)\n\n![Defect recall](experiment-defect-recall.png)\n\n"
    text += "## What the measurements show\n\n"
    best_image = max(reports, key=lambda report: report["metrics"]["image_auroc"])
    best_pixel = max(reports, key=lambda report: report["metrics"]["pixel_average_precision"] or -1)
    text += f"The highest image-AUROC point estimate belongs to `{best_image['run']}` ({best_image['metrics']['image_auroc']:.3f}); "
    text += f"the highest native-mask pixel-AP point estimate belongs to `{best_pixel['run']}` ({best_pixel['metrics']['pixel_average_precision']:.3f}). "
    text += "Image ranking and defect localization are different outcomes. The confidence intervals and fixed-threshold counts should be considered together; these test comparisons do not choose the deployed checkpoint.\n\n"
    for report in reports:
        metrics = report["metrics"]
        group_name, weakest = min(metrics["recall_by_defect"].items(), key=lambda item: item[1]["recall"])
        confusion = metrics["confusion"]
        text += f"- `{report['run']}` misses {confusion['false_negative']} defective parts and flags {confusion['false_positive']} good parts. "
        text += f"Its lowest-recall annotated group is `{group_name.split('/')[-1]}`: {weakest['detected']}/{weakest['images']} detected. "
        text += "The corresponding failure-gallery panels allow direct comparison of predicted activations with annotated defects.\n"
    text += "\nHigh precision alone cannot compensate for missed defective parts. In this defect-heavy test set, inspect recall and false alarms alongside precision. A normal operating threshold remains an empirical calibration rule rather than an accuracy guarantee.\n\n"
    text += "## Interpretation limits\n\nThese pilots differ in objective, network capacity, corruption generator, input resolution, learning rate and training budget. They provide descriptive whole-method comparisons and do not isolate the causal effect of any single component. Lower training/validation loss does not establish better real-defect detection. Pixel AP uses original masks and restored model maps; resizing can remove small input evidence. Image bootstrap intervals do not capture variation across training seeds or deployment domain shift.\n\n"
    text += "## Balanced failure galleries\n\nGallery selection is deterministic: highest-scoring false positives, lowest-scoring false negatives, and rank-spaced successful cases. Prediction activations and ground truth occupy separate panels. Display ranges are fixed per checkpoint at twice its frozen image threshold; colors are not calibrated probabilities or pixel decisions.\n\n"
    for report in reports:
        if "gallery" in report:
            text += f"- [{report['run']} gallery index](galleries/{report['run']}/index.json)\n"
            for group in ("FP", "FN", "TP", "TN"):
                selected = next((item for item in report["gallery"]["items"] if item["group"] == group), None)
                if selected:
                    text += f"\n{report['run']}: {group}, {selected['defect']}, score {selected['score']:.6g}.\n\n![{group} comparison panels](galleries/{report['run']}/{selected['file']})\n"
    text += "\nDerivative gallery images retain their dataset's CC BY-NC-SA 4.0 license and individual attribution files. Raw datasets and checkpoints remain outside Git; release assets must carry checksums and model provenance.\n"
    (output / "EXPERIMENT_REPORT.md").write_text(text)
    for report in reports:
        report.pop("history")
    (output / "experiment-metrics.json").write_text(json.dumps({"split_digest": manifest["split_digest"],
                                                               "runs": reports}, indent=2, allow_nan=False) + "\n")
    return {"report": str(output / "EXPERIMENT_REPORT.md"), "runs": [report["run"] for report in reports]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/results"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--no-galleries", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(build_report(args.runs, args.manifest, args.output, device=args.device,
                                 galleries=not args.no_galleries, bootstrap_samples=args.bootstrap_samples), indent=2))


if __name__ == "__main__":
    main()
