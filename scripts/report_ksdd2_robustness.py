"""Verify completed exploratory artifacts and render portable evidence; never infer.

Reads JSON metadata and hashes declared source/checkpoint files. It never opens
image or mask files, loads model tensors, runs inference or recalibrates weights.
No output is written until the complete two-model evidence has been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CANDIDATES = ("control", "acquisition_aug")
CONDITIONS = ("original", "gaussian_blur_radius_1", "brightness_0.8", "brightness_1.2", "jpeg_quality_60")
VALIDATION_CONDITIONS = ("clean", "brightness_0.8", "brightness_1.2", "jpeg_quality_60")
AREA_BINS = ("tiny_below_0.1_percent", "small_0.1_to_1_percent", "medium_1_to_5_percent", "large_at_least_5_percent")
STATUS = "exploratory"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inside(root, path):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path.is_relative_to(root), "Evidence path must stay within the project")
    return path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(actual, expected, name):
    if expected is None:
        require(actual is None, f"{name} must be unavailable for an empty denominator")
    else:
        require(finite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), f"{name} differs from its counts")


def wilson(successes, trials):
    if not trials:
        return None
    z = 1.959963984540054
    p, den = successes / trials, 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return [max(0., center - radius), min(1., center + radius)]


def interval_equal(actual, expected, name):
    if expected is None:
        require(actual is None, f"{name} must be unavailable")
        return
    require(isinstance(actual, list) and len(actual) == 2, f"Missing {name}")
    for got, want in zip(actual, expected):
        close(got, want, name)


def valid_interval(value, name):
    require(isinstance(value, list) and len(value) == 2 and all(finite(x) for x in value)
            and 0 <= value[0] <= value[1] <= 1, f"Invalid {name}")


def point_target(metrics):
    return metrics["defect_recall"] >= .9 and metrics["normal_false_alarm_rate"] <= .1


def verify_metrics(metrics, threshold, positives, negatives, *, native=False):
    require(metrics.get("threshold") == threshold, "A reported operating threshold changed")
    confusion = metrics["confusion"]
    require(set(confusion) == {"true_positive", "false_positive", "true_negative", "false_negative"}, "Incomplete confusion counts")
    require(all(type(v) is int and v >= 0 for v in confusion.values()), "Counts must be nonnegative integers")
    tp, fn, fp, tn = (confusion[k] for k in ("true_positive", "false_negative", "false_positive", "true_negative"))
    require(tp + fn == positives and fp + tn == negatives and metrics["images"] == positives + negatives,
            "Metrics do not cover the complete frozen test membership")
    close(metrics["defect_recall"], tp / positives, "recall")
    close(metrics["normal_false_alarm_rate"], fp / negatives, "false-alarm rate")
    close(metrics["precision"], tp / (tp + fp) if tp + fp else None, "precision")
    for key in ("image_auroc", "image_average_precision") + (("pixel_average_precision",) if native else ()):
        require(finite(metrics[key]) and 0 <= metrics[key] <= 1, f"Invalid {key}")
    uncertainty = metrics["uncertainty"]
    for key, counts in (("defect_recall_interval", (tp, positives)), ("normal_false_alarm_rate_interval", (fp, negatives)),
                        ("precision_interval", (tp, tp + fp))):
        interval_equal(uncertainty[key], wilson(*counts), key)
    require(metrics.get("measured_quality_target_met") is point_target(metrics), "Saved target flag differs from point estimates")
    if native:
        require(metrics["pixel_metric_space"] == "original", "Native-resolution pixel AP is required")
        require(uncertainty["bootstrap_samples"] == 1000 and uncertainty["normal_images"] == negatives
                and uncertainty["defective_images"] == positives and uncertainty["confidence_level"] == .95,
                "Native evaluation uncertainty method/counts differ")
        valid_interval(uncertainty["image_auroc_interval"], "image AUROC interval")
        area = metrics["defect_area_groups"]
        require(area["boundaries"] == [.001, .01, .05] and set(area["groups"]) == set(AREA_BINS), "Defect area bins differ")
        require(sum(g["images"] for g in area["groups"].values()) == positives
                and sum(g["detected"] for g in area["groups"].values()) == tp, "Defect-area counts do not match evaluation")
        for group in area["groups"].values():
            require(all(type(group[k]) is int and group[k] >= 0 for k in ("images", "detected", "missed"))
                    and group["detected"] + group["missed"] == group["images"], "Invalid area-group counts")
            close(group["recall"], group["detected"] / group["images"] if group["images"] else None, "area recall")
            interval_equal(group["recall_interval"], wilson(group["detected"], group["images"]), "area recall interval")


def verify(root):
    """Fail closed on incomplete evidence. Read no test/calibration image bytes."""
    root = Path(root).resolve()
    out, screen = root / "outputs/ksdd2-robustness-evaluation", root / "outputs/ksdd2-robustness"
    status, results, freeze = (read(out / name) for name in ("status.json", "results.json", "pre-evaluation-freeze.json"))
    require(status.get("state") == "complete" and status.get("completed") == list(CANDIDATES), "Both model evaluations must finish before reporting")
    freeze_id = digest(freeze)
    require(status.get("experimental_status") == results.get("experimental_status") == freeze.get("experimental_status") == STATUS,
            "Only explicitly exploratory evidence may be reported")
    require(status["freeze_digest"] == results["freeze_digest"] == freeze_id
            and results["selected_before_evaluation"] == "acquisition_aug" and results["thresholds_changed"] is False,
            "Completion/selection/freeze identity differs")
    require(freeze["protocol"] == "ksdd2-robustness-exploratory-evaluation-1" and freeze["pixel_space"] == "original"
            and freeze["bootstrap_samples"] == 1000 and freeze["conditions"] == list(CONDITIONS), "Evaluation method differs")
    declaration, gate, assets, screen_status, paired = (read(screen / name) for name in (
        "declaration.json", "screen-result.json", "calibrated-assets.json", "status.json", "paired-validation-uncertainty.json"))
    require(screen_status.get("training_complete") and screen_status.get("calibration_performed"), "Screen/calibration is incomplete")
    require(declaration["declaration_digest"] == digest({k: v for k, v in declaration.items() if k != "declaration_digest"})
            == freeze["training_declaration_digest"], "Training declaration digest differs")
    require(digest(gate) == freeze["gate_digest"] and gate["state"] == "awaiting_exploratory_evaluation"
            and gate["selected"] == gate["candidate"] and gate["candidate"]["candidate"] == "acquisition_aug"
            and gate["control"]["candidate"] == "control", "Prior validation selection differs")
    require(gate["rule"] == {"minimum_pooled_h_gain": .01, "maximum_each_condition_h_regression": .01},
            "The predeclared validation gate rule differs")
    gain = gate["candidate"]["pooled"]["harmonic_mean"] - gate["control"]["pooled"]["harmonic_mean"]
    close(gate["pooled_h_gain"], gain, "validation gain")
    require(gain >= .01 and gate["test_or_calibration_metrics_used_for_selection"] is False, "Validation gate did not pass")
    for condition in VALIDATION_CONDITIONS:
        change = gate["candidate"]["conditions"][condition]["harmonic_mean"] - gate["control"]["conditions"][condition]["harmonic_mean"]
        close(gate["condition_h_changes"][condition], change, "validation condition change")
        require(change >= -.01, "Validation condition regression violated the gate")
    require(declaration["maximum_new_training_runs"] == 2 and len(declaration["runs"]) == 2,
            "The declared two-run budget differs")
    matched = [{k: v for k, v in job["config"].items() if k != "candidate"} for job in declaration["runs"]]
    require(matched[0] == matched[1] and matched[0]["training_seed"] == 42
            and matched[0]["epochs"] == 100 and matched[0]["selection_every"] == 5
            and matched[0]["patience_epochs"] == 15, "The matched training budget differs")
    require([a["candidate"] for a in assets] == list(CANDIDATES) and assets == freeze["assets"], "Frozen calibrated assets differ")
    require(freeze["frozen_sources"] == declaration["source_files"], "Training/evaluation source identities differ")
    for source, expected in freeze["frozen_sources"].items():
        require(sha(inside(root, source)) == expected, f"Frozen source changed: {source}")
    for name, key in (("scripts/evaluate_ksdd2_robustness.py", "evaluation_wrapper_sha256"),
                      ("scripts/ksdd2_native_stream.py", "native_stream_evaluator_sha256")):
        require(sha(root / name) == freeze[key], f"Evaluator source changed: {name}")
    manifest_path = inside(root, declaration["inputs"]["manifest"])
    require(sha(manifest_path) == declaration["inputs"]["manifest_file_sha256"] == freeze["manifest_file_sha256"], "Manifest file changed")
    manifest = read(manifest_path)
    split = manifest["split_digest"]
    require(split == declaration["inputs"]["split_digest"] == freeze["split_digest"], "Split identity differs")
    bank_path = inside(root, declaration["inputs"]["bank"])
    bank = read(bank_path)
    require(bank["bank_digest"] == digest({k: v for k, v in bank.items() if k != "bank_digest"})
            == declaration["inputs"]["bank_digest"] and bank["split_digest"] == split, "Validation bank changed")
    test_records = manifest["splits"]["test"]  # metadata only; images/masks are never opened
    positives, negatives = sum(r["label"] == 1 for r in test_records), sum(r["label"] == 0 for r in test_records)
    require((positives, negatives) == (110, 894), "The original KSDD2 test membership differs")
    benchmark_record = next(r for r in manifest["splits"]["calibration"] if r["label"] == 0)
    benchmark = freeze["benchmark"]
    benchmark_input = {"source": benchmark_record["relative_image"], "image_sha256": benchmark_record["image_sha256"]}
    require(benchmark == {"devices": ["cpu", "mps"], "measurements": 32, "warmup": 3, "input": benchmark_input}, "Benchmark method/input identity differs")
    require(paired["original_image_groups"] == 350 and paired["bootstrap_samples"] == 1000
            and paired["used_for_gate"] is False and paired["seed"] == 4_000_042, "Paired validation uncertainty differs")
    close(paired["difference"], gate["candidate"]["pooled"]["image_auroc"] - gate["control"]["pooled"]["image_auroc"], "paired AUROC difference")
    require(len(paired["percentile_95_interval"]) == 2 and all(finite(x) for x in paired["percentile_95_interval"])
            and -1 <= paired["percentile_95_interval"][0] <= paired["percentile_95_interval"][1] <= 1, "Invalid paired validation interval")
    rows = results["results"]
    require([r["candidate"] for r in rows] == list(CANDIDATES), "Missing or reordered completed models")
    public_models, artifact_hashes = [], {}
    for path in (out / "status.json", out / "results.json", out / "pre-evaluation-freeze.json", screen / "declaration.json",
                 screen / "screen-result.json", screen / "calibrated-assets.json", screen / "paired-validation-uncertainty.json", bank_path, manifest_path):
        artifact_hashes[str(path.relative_to(root))] = sha(path)
    image_records = {r["image"]: r for r in test_records}
    image_records.update({r["relative_image"]: r for r in test_records})
    for asset, job, row in zip(assets, declaration["runs"], rows):
        name = asset["candidate"]
        run = inside(root, job["output"])
        require(job["config"]["candidate"] == name and row["checkpoint_sha256"] == asset["checkpoint_sha256"], "Candidate/model identity differs")
        selected_path, checkpoint_path = run / "selected.pt", inside(root, asset["checkpoint"])
        require(checkpoint_path == run / "checkpoint.pt" and sha(checkpoint_path) == asset["checkpoint_sha256"], "Calibrated checkpoint bytes changed")
        validation = gate["control"] if name == "control" else gate["candidate"]
        require(sha(selected_path) == validation["checkpoint_sha256"] == asset["calibration_identity"]["selected_checkpoint_sha256"]
                and asset["calibration_identity"]["gate_digest"] == digest(gate), "Selected checkpoint/gate identity differs")
        summary, calibration = read(run / "summary.json"), read(run / "calibration-summary.json")
        for value in (summary, calibration):
            require(value["config"] == job["config"] and value["split_digest"] == split
                    and value["bank_digest"] == bank["bank_digest"] and value["declaration_digest"] == declaration["declaration_digest"]
                    and value["provenance"]["robustness_source_files"] == declaration["source_files"], "Run summary provenance differs")
        require(summary["threshold"] is None and summary["best_epoch"] == validation["best_epoch"]
                and summary["completed_epochs"] == validation["completed_epochs"], "Validation-selected checkpoint summary differs")
        expected_validation = {k: v for k, v in validation.items() if k not in ("candidate", "checkpoint", "checkpoint_sha256", "best_epoch", "completed_epochs")}
        require(summary["best_validation"] == expected_validation, "Saved validation metrics differ")
        cal = calibration["calibration"]
        scores = sorted(cal["scores"])
        require(calibration["calibration_identity"] == asset["calibration_identity"] and cal["normal_count"] == len(scores) == 313
                and cal["quantile"] == .9 and cal["quantile_method"] == "linear" and cal["split"] == "calibration"
                and cal["threshold"] == calibration["threshold"] == asset["threshold"] and all(finite(x) for x in scores), "Clean-normal calibration differs")
        position = (len(scores) - 1) * .9
        low = int(position)
        close(asset["threshold"], scores[low] + (scores[low + 1] - scores[low]) * (position - low), "linear q90 threshold")
        records = {}
        for kind in ("evaluation", "stress", "latency"):
            path = inside(root, row[kind])
            require(path == out / name / f"{kind}.json", "Result path does not match its candidate")
            value = read(path)
            require(value["evaluation_freeze_digest"] == freeze_id, "Cached output belongs to another evaluation freeze")
            records[kind] = value
            artifact_hashes[str(path.relative_to(root))] = sha(path)
        evaluation, stress, latency = (records[k] for k in ("evaluation", "stress", "latency"))
        for value in (evaluation, stress):
            require(value["checkpoint_sha256"] == asset["checkpoint_sha256"] and value["threshold"] == asset["threshold"]
                    and value["split_digest"] == split and value["experimental_status"] == STATUS, "Cached model/threshold/split/status differs")
        shared_sources = {Path(path).name: value for path, value in freeze["frozen_sources"].items()
                          if path.startswith("src/inspection/")}
        extra_sources = {path: value for path, value in freeze["frozen_sources"].items()
                         if path.startswith("experiments/ksdd2_robustness/") and path.endswith(".py")}
        extra_sources["scripts/ksdd2_native_stream.py"] = freeze["native_stream_evaluator_sha256"]
        require(evaluation["evaluation_source_sha256"] == shared_sources
                and evaluation["evaluation_additional_source_sha256"] == extra_sources,
                "Cached native evaluation source hashes differ from the frozen source")
        require(evaluation["dataset"] == "KolektorSDD2" and evaluation["model_kind"] == "supervised_segmentation"
                and evaluation["training_seed"] == 42 and evaluation["calibration_normal_count"] == 313
                and evaluation["preprocess"] == job["config"]["preprocess"]
                and evaluation["validation_best_epoch"] == summary["best_epoch"]
                and evaluation["validation_best_metric"] == summary["best_validation"]["pooled"]["harmonic_mean"],
                "Cached checkpoint selection or preprocessing metadata differs")
        verify_metrics(evaluation, asset["threshold"], positives, negatives, native=True)
        require(tuple(stress["results"]) == CONDITIONS, "The five stress conditions differ")
        for metrics in stress["results"].values():
            verify_metrics(metrics, asset["threshold"], positives, negatives)
        require(stress["results"]["original"]["confusion"] == evaluation["confusion"], "Native/stress original decisions disagree")
        seen, predicted_counts = set(), dict.fromkeys(evaluation["confusion"], 0)
        for prediction in evaluation["predictions"]:
            source = image_records.get(prediction["image"])
            require(source is not None and source["relative_image"] not in seen, "Prediction membership is duplicated or outside the frozen test")
            seen.add(source["relative_image"])
            require(prediction["label"] == source["label"] and finite(prediction["score"]), "Prediction label/score differs")
            positive = prediction["score"] >= asset["threshold"]
            require(prediction["predicted_defective"] is positive, "Prediction decision does not use the frozen threshold")
            key = ("true_positive" if positive else "false_negative") if source["label"] else ("false_positive" if positive else "true_negative")
            predicted_counts[key] += 1
        require(len(seen) == len(test_records) and predicted_counts == evaluation["confusion"], "Prediction/confusion evidence is incomplete")
        require(latency["input"] == benchmark_input and set(latency["devices"]) == {"cpu", "mps"}, "Latency input/device identity differs")
        for device, measurement in latency["devices"].items():
            require(measurement["device"] == device and measurement["checkpoint_sha256"] == asset["checkpoint_sha256"]
                    and measurement["measurements"] == 32 and measurement["warmup_forwards"] == 3, "Latency method differs")
            require(all(finite(measurement[k]) and measurement[k] > 0 for k in ("single_image_median_ms", "single_image_p95_ms"))
                    and measurement["single_image_p95_ms"] >= measurement["single_image_median_ms"], "Invalid latency measurements")
        for path in (run / "summary.json", run / "calibration-summary.json", selected_path, checkpoint_path):
            artifact_hashes[str(path.relative_to(root))] = sha(path)
        gallery_dir = root / "docs/results/ksdd2-robustness-galleries" / name
        gallery_index_path, attribution_path = gallery_dir / "index.json", gallery_dir / "ATTRIBUTION.md"
        gallery = read(gallery_index_path)
        require(gallery["candidate"] == name and gallery["checkpoint_sha256"] == asset["checkpoint_sha256"]
                and gallery["experimental_status"] == STATUS and gallery["evaluation_freeze_digest"] == freeze_id
                and gallery["display_range"] == [0, 1], "Gallery model/freeze/display identity differs")
        require(gallery["groups"] == dict.fromkeys(("FP", "FN", "TP", "TN"), 3)
                and len(gallery["items"]) == 12, "Gallery must retain three examples from each decision group")
        predictions_by_image = {image_records[p["image"]]["relative_image"]: p for p in evaluation["predictions"]}
        gallery_seen, group_counts = set(), dict.fromkeys(gallery["groups"], 0)
        for item in gallery["items"]:
            require(item["source"] in predictions_by_image and item["source"] not in gallery_seen,
                    "Gallery source is duplicated or outside evaluated membership")
            gallery_seen.add(item["source"])
            p, record = predictions_by_image[item["source"]], image_records[item["source"]]
            expected_group = ("TP" if p["predicted_defective"] else "FN") if p["label"] else ("FP" if p["predicted_defective"] else "TN")
            require(item["group"] == expected_group and item["label"] == p["label"] and item["score"] == p["score"]
                    and item["threshold"] == asset["threshold"] and item["source_sha256"] == record["image_sha256"],
                    "Gallery caption differs from the saved evaluation")
            figure = inside(root, gallery_dir / item["file"])
            require(figure.parent == gallery_dir and figure.is_file(), "Gallery figure is missing or outside its folder")
            group_counts[item["group"]] += 1
        require(group_counts == gallery["groups"], "Gallery group counts differ")
        # Hash only JSON/text metadata. Figure bytes remain identified by the
        # renderer's saved checksums; this report never opens image content.
        for path in (gallery_index_path, attribution_path):
            artifact_hashes[str(path.relative_to(root))] = sha(path)
        records["gallery"] = {"index": str(gallery_index_path.relative_to(root)),
                              "attribution": str(attribution_path.relative_to(root)),
                              "index_sha256": sha(gallery_index_path), "attribution_sha256": sha(attribution_path),
                              "metadata": gallery,
                              "verification_scope": "Index/caption/source metadata checked against saved evaluation; figure bytes not reread by this reporter"}
        public_models.append({"candidate": name, "selected_for_exploratory_evaluation": name == "acquisition_aug",
                              "asset": asset, "training_summary": summary, "calibration_summary": calibration, **records})
    source_map = {r["image"]: r["relative_image"] for split_records in manifest["splits"].values() for r in split_records}

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str):
            if value in source_map:
                return source_map[value]
            if Path(value).is_absolute():
                return str(inside(root, value).relative_to(root))
        return value

    evidence = portable({"schema_version": "ksdd2-robustness-results-1", "experimental_status": STATUS,
        "training_supervision": "real defective and normal training images with pixel annotations; random initialization",
        "reported_selection": "acquisition_aug was chosen by the validation gate before exploratory evaluation",
        "published_demo": "v0.1.0 is historical and unchanged by this report; no promotion is performed",
        "quality_target": {"minimum_recall": .9, "maximum_normal_false_alarm_rate": .1,
                           "scope": "reused-test point estimates only; not an independent or population guarantee"},
        "evaluation_status": {k: v for k, v in status.items() if k != "pid"}, "evaluation_freeze": freeze,
        "training_declaration": declaration, "validation_gate": gate, "paired_validation_uncertainty": paired,
        "models": public_models, "original_local_artifact_sha256": artifact_hashes,
        "reporter_sha256": sha(Path(__file__)),
        "portability": "Machine-specific repository prefixes were replaced with relative paths, and image identifiers with manifest relative_image values. Original digests/hashes identify unmodified local records, not path-normalized copies. portable_evidence_digest covers this JSON without its own field using sorted compact JSON.",
        "limitations": ["Both training runs use seed 42; this comparison does not measure training-seed variability.",
            "350 validation source images have four paired variants; they are not 1,400 independent samples.",
            "Validation uncertainty describes reused-validation image AUROC only; it is not an interval for H or pixel AP.",
            "All reused KSDD2 test results for the new method are development-inspected/exploratory.",
            "Defect masks and examples were not opened or regenerated by this reporting script.",
            "Synthetic perturbations and image-level intervals do not establish new camera/product/batch reliability."]})
    evidence["portable_evidence_digest"] = digest(evidence)
    serialized = json.dumps(evidence, allow_nan=False)
    require(str(root) not in serialized and "/Users/" not in serialized and "/home/" not in serialized, "Absolute user path in public evidence")
    return evidence


def number(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "n/a" if value is None else f"{100 * value:.3f}%"


def ci(value, percentage=True):
    if value is None:
        return "n/a"
    scale = 100 if percentage else 1
    suffix = "%" if percentage else ""
    return f"[{value[0] * scale:.3f}, {value[1] * scale:.3f}]{suffix}"


def outcome(metrics):
    return "MET (point estimates)" if point_target(metrics) else "NOT MET"


def render(evidence):
    models, gate = evidence["models"], evidence["validation_gate"]
    selected = next(m for m in models if m["candidate"] == "acquisition_aug")
    original = selected["evaluation"]
    c = original["confusion"]
    paired = evidence["paired_validation_uncertainty"]
    interval = paired["percentile_95_interval"]
    crosses = interval[0] <= 0 <= interval[1]
    lines = ["# KolektorSDD2 acquisition robustness: exploratory results", "",
        f"**The preselected augmented model detects {c['true_positive']}/{c['true_positive'] + c['false_negative']} original-test defects and falsely flags {c['false_positive']}/{c['false_positive'] + c['true_negative']} normal images. Its original-test point target is {outcome(original)}.**",
        "This is a completed comparison of two fresh models trained with real defect masks. The candidate was selected by validation before these reused-test results were observed. Every result below is **development-inspected / exploratory**, with no new independent-success claim and no automatic model promotion. The separate published v0.1.0 model and historical evidence remain unchanged.", "",
        "[Portable evidence with full precision, predictions and hashes](KSDD2_ROBUSTNESS_RESULTS.json) · [Frozen protocol](../../experiments/ksdd2_robustness/README.md) · [Historical v0.1.0 study](CONTROLLED_STUDY.md)", "",
        "## Matched training and prior validation gate", "",
        "Control and candidate share seed 42, the original audited splits, base-16 U-Net, balanced batches of eight, Adam at 0.0003, positive-weighted BCE plus Dice, and 640-high × 256-wide letterboxing. The candidate keeps an image unchanged with probability 0.5; otherwise it applies brightness U(0.75, 1.25) followed by JPEG integer quality 60–95 before letterboxing. This tests the combined augmentation policy, not either ingredient in isolation.", "",
        "Both have a 100-epoch maximum, validation every five epochs and 15 stale training epochs of patience. Their actual stopping epochs can differ. A single pooled-selected checkpoint supplies each run's per-condition metrics; no condition-specific winning epochs are mixed together.", "",
        "| Run | Completed epochs | Selected epoch | Pooled validation AUROC | Pooled model-space pixel AP | Pooled H |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in models:
        v, s = model["training_summary"]["best_validation"], model["training_summary"]
        lines.append(f"| {model['candidate']} | {s['completed_epochs']} | {s['best_epoch']} | {number(v['pooled']['image_auroc'], 6)} | {number(v['pooled']['pixel_ap'], 6)} | {number(v['pooled']['harmonic_mean'], 6)} |")
    lines += ["", f"The frozen gate passed with pooled H gain **{gate['pooled_h_gain']:.9f}** (required ≥0.01), while every condition H regression remained within 0.01. Clean and degraded validation conditions used the same 350 source images, with four paired variants each.", "",
        "| Validation condition | Control H | Augmented H | Change |", "| --- | ---: | ---: | ---: |"]
    for condition in VALIDATION_CONDITIONS:
        lines.append(f"| {condition} | {gate['control']['conditions'][condition]['harmonic_mean']:.6f} | {gate['candidate']['conditions'][condition]['harmonic_mean']:.6f} | {gate['condition_h_changes'][condition]:+.6f} |")
    lines += ["", f"The paired grouped-bootstrap validation image-AUROC difference is **{paired['difference']:+.6f}**, with 95% interval **[{interval[0]:+.6f}, {interval[1]:+.6f}]**. " + ("It crosses zero. " if crosses else "It does not cross zero. ") + "This is descriptive uncertainty on reused validation sources, not an independent significance claim; it is not an interval for H or pixel AP and did not change the gate.",
        "JPEG validation image AUROC and pixel AP are retained separately in the JSON; an increase in their harmonic mean does not imply every component metric improved.", "",
        "## Original-test decisions and localization", "",
        "Thresholds were fit once, only after both checkpoints and the passing validation gate were frozen. Each uses the 90th percentile of the same 313 separate clean normal calibration scores. No test or stress result changed a threshold.", "",
        "| Run | TP / defects | FN | FP / normals | TN | Recall [95% Wilson] | Normal false alarms [95% Wilson] | Target |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- |"]
    for model in models:
        m = model["evaluation"]; counts = m["confusion"]; u = m["uncertainty"]
        lines.append(f"| {model['candidate']} | {counts['true_positive']}/110 | {counts['false_negative']} | {counts['false_positive']}/894 | {counts['true_negative']} | {percent(m['defect_recall'])} {ci(u['defect_recall_interval'])} | {percent(m['normal_false_alarm_rate'])} {ci(u['normal_false_alarm_rate_interval'])} | {outcome(m)} |")
    lines += ["", "| Run | Image AUROC [95% image bootstrap] | Image AP | Native pixel AP | Alert precision [95% Wilson] | Frozen threshold |",
              "| --- | --- | ---: | ---: | --- | ---: |"]
    for model in models:
        m, u = model["evaluation"], model["evaluation"]["uncertainty"]
        lines.append(f"| {model['candidate']} | {number(m['image_auroc'], 6)} {ci(u['image_auroc_interval'], False)} | {number(m['image_average_precision'], 6)} | {number(m['pixel_average_precision'], 6)} | {percent(m['precision'])} {ci(u['precision_interval'])} | {m['threshold']:.12g} |")
    lines += ["", "The joint target is recall ≥90% and normal false alarms ≤10% on the measured dataset. A passing point estimate does not establish either population-level bound; confidence intervals may cross those bounds. Image AUROC intervals use 1,000 stratified bootstrap samples; rate intervals use Wilson scores. They do not capture domain shift, training-seed variability or independent production batches. Alert precision reflects this dataset's defect prevalence.",
              "Final pixel AP uses predictions restored to original image dimensions and unchanged native masks. It is distinct from model-space validation AP. No confidence interval for pixel AP is supplied, and interpolation cannot recover details lost in resizing.", "",
              "## Native defect-size groups", "", "| Run / native annotation area | Detected / defective | Missed | Recall [95% Wilson] |", "| --- | --- | ---: | --- |"]
    for model in models:
        for name in AREA_BINS:
            group = model["evaluation"]["defect_area_groups"]["groups"][name]
            lines.append(f"| {model['candidate']} / {name} | {group['detected']}/{group['images']} | {group['missed']} | {percent(group['recall'])} {ci(group['recall_interval'])} |")
    lines += ["", "Area bins are based on the original annotation's fraction of image pixels. Empty groups have no recall estimate; small groups carry wide uncertainty. Full predictions preserve both mistakes and correct decisions. No failure image was discarded by this report.", "",
              "## Auditable examples and errors", "",
              "The two galleries contain 24 real dataset examples in total: three false positives, false negatives, true positives and true negatives per model. False positives use the highest scores, false negatives the lowest, and correct decisions evenly spaced score ranks. This stated selection is illustrative, not a representative sample or a substitute for all 1,004 predictions.", "",
              "- [Control gallery index](ksdd2-robustness-galleries/control/index.json) · [Control attribution](ksdd2-robustness-galleries/control/ATTRIBUTION.md)",
              "- [Augmented gallery index](ksdd2-robustness-galleries/acquisition_aug/index.json) · [Augmented attribution](ksdd2-robustness-galleries/acquisition_aug/ATTRIBUTION.md)", "",
              "All maps share the fixed 0–1 color range. The single-image maps use the same frozen weights; captions and decisions retain the saved batched evaluation scores. Very small nonzero responses can be visually dark on this range even when they exceed the much smaller operating threshold. A dark map is not a numerical zero or a calibrated defect probability.", "",
              "Augmented model: an annotated-normal image incorrectly flagged as defective:", "",
              "![Augmented model false positive on KolektorSDD2 test/20358.png](ksdd2-robustness-galleries/acquisition_aug/FP_0358.png)", "",
              "Augmented model: a real defect missed at the unchanged threshold:", "",
              "![Augmented model false negative on KolektorSDD2 test/20159.png](ksdd2-robustness-galleries/acquisition_aug/FN_0159.png)", "",
              "## Paired stress conditions at unchanged thresholds", "",
              "| Run / condition | Detected / defects | False alarms / normals | Recall [95% Wilson] | False alarms [95% Wilson] | Decisions changed | Target |",
              "| --- | --- | --- | --- | --- | ---: | --- |"]
    failures = []
    for model in models:
        for condition in CONDITIONS:
            m = model["stress"]["results"][condition]; c = m["confusion"]; u = m["uncertainty"]
            lines.append(f"| {model['candidate']} / {condition} | {c['true_positive']}/110 | {c['false_positive']}/894 | {percent(m['defect_recall'])} {ci(u['defect_recall_interval'])} | {percent(m['normal_false_alarm_rate'])} {ci(u['normal_false_alarm_rate_interval'])} | {m['decision_flips']} | {outcome(m)} |")
            if not point_target(m):
                failures.append(f"{model['candidate']}/{condition}")
    lines += ["", ("Point targets remain unmet for: " + ", ".join(f"`{name}`" for name in failures) + ".") if failures else "Both models meet the point targets on all five measured conditions; this still does not establish independent reliability.",
              "The five conditions reuse the same test images, without retraining or recalibration. Their observations are paired, not independent datasets. Stress results measure image decisions only; stress pixel AP was not computed. These specified blur/brightness/JPEG transformations do not establish robustness to arbitrary cameras, lighting, products or new defect families.", "",
              "## Local CPU and MPS timings", "", "| Run | Device | Warm median (ms) | Warm p95 (ms) | Measurements / warmups |", "| --- | --- | ---: | ---: | --- |"]
    for model in models:
        for device in ("cpu", "mps"):
            t = model["latency"]["devices"][device]
            lines.append(f"| {model['candidate']} | {device} | {t['single_image_median_ms']:.3f} | {t['single_image_p95_ms']:.3f} | {t['measurements']} / {t['warmup_forwards']} |")
    lines += ["", "Benchmarks ran sequentially under the project scheduler lock, using the same checksum-pinned first clean normal calibration image. They measure warm forward/scoring time, excluding loading, decoding, preprocessing, transfers, rendering and hosting overhead. The lock serializes this project's jobs; it cannot enforce external system idleness. These are local measurements, not hosted response-time guarantees.", "",
              "## Provenance and limits", "", "| Identity | Digest |", "| --- | --- |",
              f"| Training declaration | `{evidence['training_declaration']['declaration_digest']}` |",
              f"| Evaluation freeze | `{evidence['evaluation_status']['freeze_digest']}` |",
              f"| Audited split | `{evidence['evaluation_freeze']['split_digest']}` |"]
    for model in models:
        lines.append(f"| {model['candidate']} calibrated checkpoint | `{model['asset']['checkpoint_sha256']}` |")
    lines += ["", "The portable JSON preserves all model, calibration, source and evaluator identities, original local artifact hashes, full predictions and uncertainty. Machine-specific path prefixes are removed; original hashes identify original artifacts, not the normalized copies. The report reads saved artifacts only and never reruns models, opens dataset images, changes thresholds or promotes a checkpoint.",
              "Both models use random initialization and real-defect supervision in the separate KolektorSDD2 scenario. They do not establish success for the normal-only MVTec parts. Both runs use seed 42, so training-seed variability is unmeasured. The original KSDD2 test/stress results informed this new method, making all reused results exploratory. A genuinely untouched same-category holdout and independent acquisition/product groups are still needed for new independent reliability claims.", ""]
    return "\n".join(lines)


def publish(root, output="docs/results"):
    root = Path(root).resolve()
    evidence = verify(root)
    directory = inside(root, output)
    payloads = {"KSDD2_ROBUSTNESS_RESULTS.json": json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
                "KSDD2_ROBUSTNESS_RESULTS.md": render(evidence)}
    # Check every existing destination before writing either artifact.
    for name, body in payloads.items():
        path = directory / name
        require(not path.exists() or path.read_text() == body, f"Refusing to replace different published evidence: {name}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in payloads.items():
        path = directory / name
        if not path.exists():
            temporary = path.with_name(path.name + ".tmp")
            require(not temporary.exists(), "A prior report staging file needs inspection")
            temporary.write_text(body)
            temporary.replace(path)
    return {"status": "verified_and_rendered", "experimental_status": STATUS,
            "outputs": [str((directory / name).relative_to(root)) for name in payloads],
            "portable_evidence_digest": evidence["portable_evidence_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default="docs/results")
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.output), indent=2))
