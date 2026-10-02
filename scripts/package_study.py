"""Prepare a local, reviewable release candidate from a completed frozen study.

This never publishes assets, mutates Git, or replaces artifacts/models.json.
Examples are copied only after the complete study passes its evidence gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    # Direct script execution puts scripts/, rather than the project root, on
    # sys.path. Make the same imports work for both the CLI and module usage.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_study import quality_target_met, select_by_validation

VERSION = "v0.1.0"
CATEGORIES = ("metal_nut", "screw", "transistor")
OWN_KINDS = {"joint_reconstruction_segmentation", "segmentation_only", "supervised_segmentation"}
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
DATASETS = {
    "MVTec AD": {"url": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
                 "credit": "MVTec Software GmbH; Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger (CVPR 2019)"},
    "KolektorSDD2": {"url": "https://www.vicos.si/resources/kolektorsdd2/",
                    "credit": "Kolektor Group; Jakob Božič, Domen Tabernik and Danijel Skočaj (Computers in Industry, 2021)"},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def project_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Study artifact path escapes the project root")
    return path


def verify_completed_study(root, study):
    """Use the reporting agent's shared, metadata-only evidence verifier."""
    from scripts.report_study import verify_study
    return verify_study(root, study=study)


def _selection(study, name, expected):
    choice = read_json(study / "selections" / f"{name}.json")
    measurements = {item["checkpoint_sha256"]: item["cpu_p50_ms"] for item in choice["tie_measurements"]}
    replay = select_by_validation(choice["candidates"], lambda item: measurements[item["checkpoint_sha256"]])
    if any(choice.get(key) != value for key, value in replay.items()) or choice["selected"] != expected:
        raise ValueError("Deployment differs from its pretest validation-only selection")


def _verified_inputs(root, study):
    """Reject incomplete evidence before opening weights or dataset images."""
    status = read_json(study / "status.json")
    if status.get("state") != "complete" or status.get("protocol") != "study-v2":
        raise ValueError("A complete study-v2 is required before release packaging")
    verified = verify_completed_study(root, study.relative_to(root).as_posix())
    results = read_json(study / "study-results.json")
    if results.get("protocol") != "study-v2" or results.get("model_and_seed_selection_used_test_metrics") is not False:
        raise ValueError("Release evidence must preserve validation-only model and seed selection")
    freeze = read_json(study / "mvtec-pretest-freeze.json")
    deployments = results["pretest_selections"]
    if set(deployments) != set(CATEGORIES) or freeze["deployments"] != deployments or freeze.get("test_selection") is not False:
        raise ValueError("MVTec deployment map differs from the pretest freeze")
    evaluations = dict(results["all_mvtec_evaluations"])
    freezes = [freeze]
    fallback = results["conditional_supervised"]
    expected_fallback = not any(quality_target_met(evaluations[item["checkpoint_sha256"]]) for item in deployments.values())
    if fallback.get("triggered") is not expected_fallback:
        raise ValueError("Conditional supervised trigger differs from frozen MVTec targets")
    if fallback["triggered"]:
        supervised = read_json(study / "ksdd2-pretest-freeze.json")
        selected = fallback["selected"]
        if supervised["deployments"] != {"kolektor_surface": selected} or supervised.get("test_selection") is not False:
            raise ValueError("Supervised deployment differs from its pretest freeze")
        deployments = {**deployments, "kolektor_surface": selected}
        evaluations.update(fallback["evaluations"])
        freezes.append(supervised)
    for frozen in freezes:
        for item in frozen["checkpoints"]:
            if sha256(project_path(root, item["checkpoint"])) != item["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint bytes changed")
            metric = evaluations[item["checkpoint_sha256"]]
            if (metric["checkpoint_sha256"] != item["checkpoint_sha256"]
                    or metric["split_digest"] != item["split_digest"]
                    or metric["threshold"] != item["threshold"]):
                raise ValueError("Frozen evaluation hash, split or threshold differs")
            evaluation_path = study / "evaluations" / f"{Path(item['run']).name}.json"
            if read_json(evaluation_path) != metric:
                raise ValueError("Study results and saved evaluation disagree")
    payloads = {}
    import torch
    for category, item in deployments.items():
        name = "deployment-ksdd2" if category == "kolektor_surface" else f"deployment-{category}"
        _selection(study, name, item)
        if item["model_kind"] not in OWN_KINDS:
            raise ValueError("Only project-owned random-initialized deployment models may enter this release")
        checkpoint = torch.load(project_path(root, item["checkpoint"]), map_location="cpu", weights_only=True)
        if (checkpoint.get("resumable") is not False or checkpoint.get("model_kind") != item["model_kind"]
                or checkpoint.get("split_digest") != item["split_digest"]
                or checkpoint.get("threshold") != item["threshold"]
                or checkpoint.get("best_metric") != item["validation_score"]
                or checkpoint.get("config") != item["config"]):
            raise ValueError("Deployment checkpoint metadata differs from its pretest candidate")
        if not math.isfinite(item["threshold"]):
            raise ValueError("A finite frozen threshold is required")
        payloads[category] = checkpoint
    return results, deployments, evaluations, payloads, verified


def choose_examples(manifest, category):
    """First relative path in each defect family, independent of model scores."""
    root = Path(manifest["root"]).resolve()
    groups = {}
    for record in manifest["splits"]["test"]:
        if record["category"] != category:
            continue
        relative = Path(record["image"]).resolve().relative_to(root).as_posix()
        key = (record["defect"], record["label"])
        if key not in groups or relative < groups[key][0]:
            groups[key] = (relative, record)
    if not groups or not any(key[1] == 0 for key in groups) or not any(key[1] == 1 for key in groups):
        raise ValueError("Examples require both normal and defective original test records")
    return sorted(groups.values(), key=lambda item: (item[1]["label"], item[0]))


def _copy_examples(root, destination, category, candidate):
    from inspection.evaluate import load_evaluation_manifest
    manifest = load_evaluation_manifest(project_path(root, candidate["manifest"]), verify_splits=())
    if manifest["split_digest"] != candidate["split_digest"]:
        raise ValueError("Example manifest differs from the frozen deployment split")
    dataset = "KolektorSDD2" if category == "kolektor_surface" else "MVTec AD"
    attribution = DATASETS[dataset]
    examples = []
    for index, (relative, record) in enumerate(choose_examples(manifest, category)):
        defect = re.sub(r"[^a-zA-Z0-9_-]", "_", record["defect"])
        source = Path(record["image"])
        target = Path("assets/examples") / f"{category}_{index:02d}_{defect}{source.suffix.lower()}"
        if sha256(source) != record["image_sha256"]:
            raise ValueError("Audited example bytes changed")
        (destination / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination / target)
        if sha256(destination / target) != record["image_sha256"]:
            raise ValueError("Copied example checksum differs")
        examples.append({"label": "Normal part" if not record["label"] else record["defect"].replace("_", " ").capitalize(),
                         "path": target.as_posix(), "defective": bool(record["label"]),
                         "sha256": record["image_sha256"], "source_path": relative,
                         "source_split": "original test", "dataset": dataset, "source_url": attribution["url"],
                         "license": "CC BY-NC-SA 4.0", "license_url": LICENSE_URL, "changes": "none; byte-identical copy",
                         "attribution": f"{dataset} · {relative} · {attribution['credit']} · CC BY-NC-SA 4.0"})
    return examples


def _model_summary(category, candidate, metric, checkpoint, parameter_count):
    supervised = category == "kolektor_surface"
    return {"category": category, "checkpoint_sha256": candidate["checkpoint_sha256"],
            "model_kind": candidate["model_kind"], "training_seed": candidate["seed"],
            "initialization": "random; no pretrained weights", "split_digest": candidate["split_digest"],
            "validation_score": candidate["validation_score"], "best_epoch": candidate["best_epoch"],
            "model_selection": "pretest validation harmonic mean; CPU latency tie-break; never test-winning seed",
            "supervision": "real defect images and masks" if supervised else "normal images and procedural corruption masks",
            "experimental_status": "exploratory" if category == "metal_nut" else "frozen-held-out",
            "preprocess": checkpoint.get("preprocess", {"mode": "square", "height": checkpoint.get("image_size"), "width": checkpoint.get("image_size")}),
            "parameters": parameter_count, "threshold": candidate["threshold"], "decision_rule": ">=",
            "score_definition": checkpoint.get("score_definition", "mean of top 1% sigmoid segmentation pixels; not a probability"),
            "calibration": checkpoint.get("calibration"), "config": candidate["config"],
            "provenance": checkpoint.get("provenance"),
            "pixel_metric_space": metric.get("pixel_metric_space", "original"),
            "metrics": {key: metric.get(key) for key in ("image_auroc", "image_average_precision", "pixel_average_precision", "defect_recall", "normal_false_alarm_rate", "precision", "confusion", "uncertainty", "recall_by_defect", "defect_area_groups")},
            "measured_quality_target_met": quality_target_met(metric),
            "quality_target": {"defect_recall_minimum": .90, "normal_false_alarm_rate_maximum": .10},
            "limits": "Dataset point estimates only. Metal-nut development is exploratory. Small calibration sets, training-seed variability and domain shift limit generalization; this is not production approval."}


def _copy_galleries(root, destination, verified, categories):
    sources = verified["gallery_sources"]
    if set(sources) != set(categories):
        raise ValueError("Verified selected galleries do not cover every deployment")
    for category, folder in sources.items():
        folder = project_path(root, folder)
        index = read_json(folder / "index.json")
        files = {"index.json", "ATTRIBUTION.md", *(item["file"] for item in index["items"])}
        target_dir = destination / "controlled-study-galleries" / category
        target_dir.mkdir(parents=True)
        for filename in sorted(files):
            source = (folder / filename).resolve()
            if not source.is_relative_to(folder):
                raise ValueError("Gallery panel path escapes its verified directory")
            key = source.relative_to(root).as_posix()
            expected = verified["evidence_sha256"].get(key)
            if expected is None or sha256(source) != expected:
                raise ValueError("Gallery source is absent from verified evidence or changed")
            target = target_dir / source.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != expected:
                raise ValueError("Copied gallery checksum differs")
        reference = verified["portable"]["categories"][category]["gallery"]
        expected_reference = f"controlled-study-galleries/{category}/index.json"
        if reference != expected_reference or not (destination / reference).is_file():
            raise ValueError("Portable gallery reference does not resolve inside the release bundle")


def package_study(root, study="outputs/study-v2", output="outputs/release-v0.1.0"):
    root = Path(root).resolve()
    study = project_path(root, study)
    destination = project_path(root, output)
    if destination.exists():
        raise FileExistsError("Release destination already exists; existing review bundles are never replaced")
    results, deployments, evaluations, checkpoints, verified = _verified_inputs(root, study)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        models, summaries = [], []
        for category, candidate in deployments.items():
            metric = evaluations[candidate["checkpoint_sha256"]]
            checkpoint = checkpoints[category]
            weight_path = Path("weights") / f"{category}-seed{candidate['seed']}.pt"
            source = project_path(root, candidate["checkpoint"])
            if source.stat().st_size > 100 * 1024 * 1024:
                raise ValueError("Own model exceeds the 100 MiB release-asset limit")
            (staging / weight_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, staging / weight_path)
            if sha256(staging / weight_path) != candidate["checkpoint_sha256"]:
                raise ValueError("Copied weights differ from the pretest frozen checkpoint")
            examples = _copy_examples(root, staging, category, candidate)
            detail = verified["portable"]["categories"][category]["selected"]
            parameters = detail["parameters"]
            if (not isinstance(parameters, int) or isinstance(parameters, bool) or parameters <= 0
                    or detail["checkpoint_sha256"] != candidate["checkpoint_sha256"]):
                raise ValueError("Verified parameter count belongs to an invalid or different deployment")
            summary = _model_summary(category, candidate, metric, checkpoint, parameters)
            summary["training_seed_variability"] = (results["conditional_supervised"].get("training_seed_variability")
                if category == "kolektor_surface" else results.get("mvtec_training_seed_variability", {}).get(category))
            summaries.append(summary)
            write_json(staging / "models" / f"{category}.json", summary)
            scope = "exploratory" if category == "metal_nut" else "held-out"
            passed = summary["measured_quality_target_met"]
            models.append({"id": f"{category.replace('_', '-')}-study-v2-seed{candidate['seed']}",
                           "label": f"{category.replace('_', ' ').title()} · {scope}", "category": category,
                           "status_label": f"{scope.upper()} MODEL · DATASET TARGET {'MET' if passed else 'NOT MET'}",
                           "description": "Compact segmentation model trained from random weights; see the bundled model summary.",
                           "conditions": "Inputs must match this dataset category and acquisition conditions; category is selected by the user, not recognized by the model.",
                           "local_path": weight_path.as_posix(), "sha256": candidate["checkpoint_sha256"],
                           "metrics": summary["metrics"], "threshold": candidate["threshold"],
                           "model_kind": candidate["model_kind"], "training_seed": candidate["seed"],
                           "experimental_status": summary["experimental_status"], "target_met": passed,
                           "evaluation_note": f"Pretest validation-selected deployment; {scope} test results. Frozen 90th-percentile normal calibration threshold. Scores are not probabilities; dataset results do not guarantee production reliability.",
                           "examples": examples})
        write_json(staging / "artifacts/models.json", {"schema_version": 1, "models": models,
                   "release_candidate": True, "published": False, "path_base": "release bundle root",
                   "version": VERSION, "registry_installation": "Requires deliberate review and path adjustment or moving bundled assets; does not replace the repository registry."})
        write_json(staging / "study-summary.json", {"protocol": "study-v2", "version": VERSION,
                   "selected_models": summaries, "conditional_supervised_triggered": results["conditional_supervised"]["triggered"],
                   "any_validation_selected_model_met_dataset_target": results["any_validation_selected_model_met_dataset_target"],
                   "dataset_fallback_trigger_uses_frozen_mvtec_test_targets": True,
                   "pretrained_reference": "Separate comparison only; no pretrained reference weights or registry entries packaged."})
        evidence_names = ["status.json", "declaration.json", "study-results.json", "mvtec-pretest-freeze.json"]
        if results["conditional_supervised"]["triggered"]:
            evidence_names.append("ksdd2-pretest-freeze.json")
        evidence = {name: sha256(study / name) for name in evidence_names}
        for directory in ("selections", "evaluations", "analyses"):
            evidence.update({path.relative_to(study).as_posix(): sha256(path)
                             for path in sorted((study / directory).rglob("*.json"))})
        write_json(staging / "evidence.json", {"source_study": study.relative_to(root).as_posix(),
                   "source_evidence_sha256": evidence, "packager_sha256": sha256(__file__),
                   "shared_verified_evidence_sha256": verified["evidence_sha256"],
                   "deployment_weight_hashes": {category: item["checkpoint_sha256"] for category, item in deployments.items()}})
        write_json(staging / "controlled-study.json", verified["portable"])
        _copy_galleries(root, staging, verified, deployments)
        attribution_lines = ["# Example image attribution", "", "Examples are unmodified, byte-identical dataset copies. Runtime overlays are adaptations and retain the dataset license.", "", "Selection: lexicographically first original test image in every defect family, including normal; model scores are not used.", ""]
        for model in models:
            for example in model["examples"]:
                attribution_lines.append(f"- `{example['path']}`: {example['attribution']}. [Source]({example['source_url']}); [license]({LICENSE_URL}). Changes: none.")
        (staging / "assets/examples/ATTRIBUTION.md").write_text("\n".join(attribution_lines) + "\n")
        notes = [f"# Industrial Inspection {VERSION} — local release candidate", "",
                 "This bundle is prepared for review. It has not been published and does not replace the demo's active model registry.", "",
                 "All included models were trained from random initialization. Deployments were selected using pretest validation evidence, not the best test seed. The ImageNet-pretrained PatchCore comparison remains separate.", "",
                 "| Category | Evaluation status | Training seed | Defect recall | Normal false alarms | Dataset target |", "|---|---|---:|---:|---:|---|"]
        for summary in summaries:
            metric = summary["metrics"]
            notes.append(f"| {summary['category']} | {summary['experimental_status']} | {summary['training_seed']} | {metric['defect_recall']:.3f} | {metric['normal_false_alarm_rate']:.3f} | {'met' if summary['measured_quality_target_met'] else 'NOT MET'} |")
        notes += ["", "The target requires defect recall ≥0.90 and normal false alarms ≤0.10 on the dataset test. Passing this descriptive target is not production approval. Metal-nut test results are exploratory because that category was inspected during development. Screw, transistor and conditional KolektorSDD2 retain their declared held-out status.", "",
                  "KolektorSDD2, if included, is a separate task with real-defect supervision. Its fallback was triggered by frozen MVTec test targets; model and seed selection still use validation only. Metrics across these datasets are not interchangeable.", "",
                  "Weights preserve the exact frozen checkpoint bytes. SHA256SUMS covers all bundle files except itself. Check model summaries for preprocessing, calibration, provenance and uncertainty. Scores and heatmaps are not calibrated probabilities.", "",
                  "The candidate artifacts/models.json uses paths relative to this bundle root and deliberately has no download URLs. Review the bundle, checksum files, licensing and path installation before explicitly updating the repository registry or publishing release assets.", "",
                  "Source-code license: MIT. Dataset examples retain CC BY-NC-SA 4.0, attribution, noncommercial and share-alike conditions; see assets/examples/ATTRIBUTION.md."]
        (staging / "RELEASE_NOTES.md").write_text("\n".join(notes) + "\n")
        if (root / "LICENSE").is_file():
            shutil.copyfile(root / "LICENSE", staging / "SOURCE_LICENSE")
        sums = [f"{sha256(path)}  {path.relative_to(staging).as_posix()}" for path in sorted(staging.rglob("*")) if path.is_file()]
        (staging / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        for path, digest in verified["evidence_sha256"].items():
            if sha256(project_path(root, path)) != digest:
                raise ValueError("Verified study evidence changed during release packaging")
        if destination.exists():
            raise FileExistsError("Release destination appeared during packaging; refusing to replace it")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--study", default="outputs/study-v2")
    parser.add_argument("--output", default="outputs/release-v0.1.0")
    args = parser.parse_args()
    print(package_study(args.root, args.study, args.output))


if __name__ == "__main__":
    main()
