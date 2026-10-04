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

CANDIDATES = ("control", "decision_head")
CONDITIONS = ("original", "gaussian_blur_radius_1", "brightness_0.8", "brightness_1.2", "jpeg_quality_60")
VALIDATION_CONDITIONS = ("clean", "brightness_0.8", "brightness_1.2", "jpeg_quality_60")
AREA_BINS = ("tiny_below_0.1_percent", "small_0.1_to_1_percent", "medium_1_to_5_percent", "large_at_least_5_percent")
STATUS = "exploratory"
IDENTITIES = {
    "control": ("supervised_segmentation", "mean highest 1% sigmoid segmentation pixels within valid letterbox area; not a calibrated probability"),
    "decision_head": ("supervised_segmentation_detached_decision_v1", "raw detached-context classification logit; greater means more defective; not a calibrated probability"),
}
GATE_RULE = {"minimum_positive_pauc_gain": True, "minimum_pauc_shortfall_reduction": .1,
             "maximum_each_condition_pauc_regression": .01, "maximum_pooled_pixel_ap_regression": .01,
             "maximum_each_condition_pixel_ap_regression": .01}


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


def verify_predictions(metrics, image_records, threshold):
    """Recompute decisions, confusion and image ranking metrics from saved scores."""
    seen, confusion = {}, dict.fromkeys(("true_positive", "false_positive", "true_negative", "false_negative"), 0)
    for prediction in metrics["predictions"]:
        source = image_records.get(prediction["image"])
        require(source is not None and source["relative_image"] not in seen, "Duplicated or foreign prediction membership")
        require(prediction["label"] == source["label"] and finite(prediction["score"]), "Invalid prediction label/score")
        positive = prediction["score"] >= threshold
        require(prediction["predicted_defective"] is positive, "Prediction decision differs from frozen score threshold")
        seen[source["relative_image"]] = prediction
        key = ("true_positive" if positive else "false_negative") if source["label"] else ("false_positive" if positive else "true_negative")
        confusion[key] += 1
    require(len(seen) == metrics["images"] == 1004 and confusion == metrics["confusion"], "Prediction confusion/membership differs")
    positives = confusion["true_positive"] + confusion["false_negative"]
    negatives = confusion["false_positive"] + confusion["true_negative"]
    groups = {}
    for row in seen.values():
        counts = groups.setdefault(row["score"], [0, 0])
        counts[row["label"]] += 1
    negatives_below, favorable = 0, 0.
    for score in sorted(groups):
        neg, pos = groups[score]
        favorable += pos * (negatives_below + neg / 2)
        negatives_below += neg
    close(metrics["image_auroc"], favorable / (positives * negatives), "image AUROC from saved predictions")
    cumulative_positive, cumulative_count, ap = 0, 0, 0.
    for score in sorted(groups, reverse=True):
        neg, pos = groups[score]
        cumulative_positive += pos
        cumulative_count += neg + pos
        ap += (pos / positives) * (cumulative_positive / cumulative_count)
    close(metrics["image_average_precision"], ap, "image AP from saved predictions")
    return seen


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
    out, screen = root / "outputs/ksdd2-decision-evaluation", root / "outputs/ksdd2-decision"
    status, results, freeze = (read(out / name) for name in ("status.json", "results.json", "pre-evaluation-freeze.json"))
    require(status.get("state") == "complete" and status.get("completed") == list(CANDIDATES), "Both model evaluations must finish before reporting")
    freeze_id = digest(freeze)
    require(status.get("experimental_status") == results.get("experimental_status") == freeze.get("experimental_status") == STATUS,
            "Only explicitly exploratory evidence may be reported")
    require(status["freeze_digest"] == results["freeze_digest"] == freeze_id
            and results["selected_before_evaluation"] == "decision_head" and results["thresholds_changed"] is False,
            "Completion/selection/freeze identity differs")
    require(freeze["protocol"] == "ksdd2-decision-exploratory-evaluation-1" and freeze["pixel_space"] == "original"
            and freeze["bootstrap_samples"] == 1000 and freeze["conditions"] == list(CONDITIONS), "Evaluation method differs")
    require(freeze["score_contracts"] == {name: dict(zip(("model_kind", "score_definition"), values))
                                         for name, values in IDENTITIES.items()}, "Frozen score contracts differ")
    declaration, gate, assets, screen_status, paired = (read(screen / name) for name in (
        "declaration.json", "screen-result.json", "calibrated-assets.json", "status.json", "paired-validation-uncertainty.json"))
    require(screen_status.get("training_complete") and screen_status.get("calibration_performed"), "Screen/calibration is incomplete")
    require(declaration["declaration_digest"] == digest({k: v for k, v in declaration.items() if k != "declaration_digest"})
            == freeze["training_declaration_digest"], "Training declaration digest differs")
    require(digest(gate) == freeze["gate_digest"] and gate["state"] == "awaiting_exploratory_evaluation"
            and gate["selected"] == gate["candidate"] and gate["candidate"]["candidate"] == "decision_head"
            and gate["control"]["candidate"] == "control", "Prior validation selection differs")
    require(gate["rule"] == declaration["gate"] == GATE_RULE,
            "The predeclared validation gate rule differs")
    gain = gate["candidate"]["pooled"]["image_pauc"] - gate["control"]["pooled"]["image_pauc"]
    shortfall = 1 - gate["control"]["pooled"]["image_pauc"]
    close(gate["pooled_pauc_gain"], gain, "validation gain")
    close(gate["control_pauc_shortfall"], shortfall, "control partial-AUROC shortfall")
    close(gate["pauc_shortfall_reduction"], gain / shortfall if shortfall > 0 else None, "partial-AUROC shortfall reduction")
    pixel_gain = gate["candidate"]["pooled"]["pixel_ap"] - gate["control"]["pooled"]["pixel_ap"]
    close(gate["pooled_pixel_ap_change"], pixel_gain, "pooled validation pixel-AP change")
    require(gain > 0 and gain >= .1 * shortfall and pixel_gain >= -.01
            and gate["test_or_calibration_metrics_used_for_selection"] is False
            and gate["further_training_allowed_by_this_screen"] is False, "Validation gate did not pass")
    for condition in VALIDATION_CONDITIONS:
        for metric in ("image_pauc", "pixel_ap"):
            change = gate["candidate"]["conditions"][condition][metric] - gate["control"]["conditions"][condition][metric]
            close(gate["condition_changes"][condition][metric], change, "validation condition change")
            require(change >= -.01, "Validation condition regression violated the gate")
    require(declaration["maximum_new_training_runs"] == 2 and len(declaration["runs"]) == 2,
            "The declared two-run budget differs")
    matched = [{k: v for k, v in job["config"].items() if k != "candidate"} for job in declaration["runs"]]
    require(matched[0] == matched[1] and matched[0]["training_seed"] == 42
            and matched[0]["epochs"] == 100 and matched[0]["selection_every"] == 5
            and matched[0]["patience_epochs"] == 15 and matched[0]["max_fpr"] == .1
            and matched[0]["classification_gradients_to_segmentation"] is False,
            "The matched training budget or scoring policy differs")
    require([a["candidate"] for a in assets] == list(CANDIDATES) and assets == freeze["assets"], "Frozen calibrated assets differ")
    require(freeze["frozen_sources"] == declaration["source_files"], "Training/evaluation source identities differ")
    for source, expected in freeze["frozen_sources"].items():
        require(sha(inside(root, source)) == expected, f"Frozen source changed: {source}")
    for source, expected in declaration["prior_evidence"].items():
        require(sha(inside(root, source)) == expected, f"Prior evidence changed: {source}")
    prior_report = read(root / "docs/results/KSDD2_ROBUSTNESS_RESULTS.json")
    reference_sha = prior_report["evaluation_freeze"]["native_stream_evaluator_sha256"]
    require(sha(root / "scripts/ksdd2_native_stream.py") == reference_sha == freeze["native_stream_reference_sha256"],
            "Prior native evaluator reference changed")
    for name, key in (("scripts/evaluate_ksdd2_decision.py", "evaluation_wrapper_sha256"),
                      ("scripts/ksdd2_decision_native.py", "native_stream_evaluator_sha256"),
                      ("scripts/ksdd2_decision_inference.py", "inference_adapter_sha256")):
        require(sha(root / name) == freeze[key], f"Evaluator source changed: {name}")
    manifest_path = inside(root, declaration["inputs"]["manifest"])
    require(sha(manifest_path) == declaration["inputs"]["manifest_file_sha256"] == freeze["manifest_file_sha256"], "Manifest file changed")
    manifest = read(manifest_path)
    split = manifest["split_digest"]
    require(split == declaration["inputs"]["split_digest"] == freeze["split_digest"], "Split identity differs")
    bank_path = inside(root, declaration["inputs"]["bank"])
    bank = read(bank_path)
    require(sha(bank_path) == declaration["inputs"]["bank_file_sha256"]
            == declaration["inputs"]["source_bank_file_sha256"]
            == sha(inside(root, declaration["inputs"]["source_bank"])), "Copied validation bank bytes differ")
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
            and paired["used_for_gate"] is False and paired["seed"] == 5_000_042, "Paired validation uncertainty differs")
    close(paired["difference"], gain, "paired partial-AUROC difference")
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
        model_kind, score_definition = IDENTITIES[name]
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
                    and value["provenance"]["decision_source_files"] == declaration["source_files"], "Run summary provenance differs")
            require(value["model_kind"] == model_kind and value["score_definition"] == score_definition
                    and value["decision_model_config"] == {"candidate": name, "base_channels": 16, "head_seed": 9_000_042},
                    "Run model/score identity differs")
            expected_head_parameters = 25747 if name == "decision_head" else 0
            require(value["parameter_counts"] == {"segmentation": 488705, "decision_head": expected_head_parameters,
                                                  "total": 488705 + expected_head_parameters},
                    "Declared parameter counts differ")
        require(summary["threshold"] is None and summary["best_epoch"] == validation["best_epoch"]
                and summary["completed_epochs"] == validation["completed_epochs"], "Validation-selected checkpoint summary differs")
        expected_validation = {k: v for k, v in validation.items() if k not in ("candidate", "checkpoint", "checkpoint_sha256", "best_epoch", "completed_epochs")}
        require(summary["best_validation"] == expected_validation, "Saved validation metrics differ")
        history_path = run / "history.json"
        history = read(history_path)
        eligible = [entry for entry in history if entry["validation"] is not None]
        best = max(eligible, key=lambda entry: entry["validation"]["pooled"]["image_pauc"])
        require(best["epoch"] == summary["best_epoch"] and best["validation"] == summary["best_validation"]
                and len(history) == summary["completed_epochs"] and summary["stale_epochs"] == 15,
                "History differs from earliest-best partial-AUROC selection and stopping")
        cal = calibration["calibration"]
        scores = sorted(cal["scores"])
        require(calibration["calibration_identity"] == asset["calibration_identity"] and cal["normal_count"] == len(scores) == 313
                and cal["quantile"] == .9 and cal["quantile_method"] == "linear" and cal["split"] == "calibration"
                and cal["threshold"] == calibration["threshold"] == asset["threshold"] and all(finite(x) for x in scores)
                and cal["model_kind"] == model_kind and cal["score_definition"] == score_definition,
                "Clean-normal calibration or score identity differs")
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
                    and value["split_digest"] == split and value["experimental_status"] == STATUS
                    and value["model_kind"] == model_kind and value["score_definition"] == score_definition,
                    "Cached model/score/threshold/split/status differs")
        shared_sources = {Path(path).name: value for path, value in freeze["frozen_sources"].items()
                          if path.startswith("src/inspection/")}
        extra_sources = dict(freeze["frozen_sources"])
        extra_sources["scripts/ksdd2_decision_native.py"] = freeze["native_stream_evaluator_sha256"]
        extra_sources["scripts/ksdd2_decision_inference.py"] = freeze["inference_adapter_sha256"]
        extra_sources["scripts/ksdd2_native_stream.py"] = reference_sha
        for value in (evaluation, stress):
            require(value["evaluation_source_sha256"] == shared_sources
                    and value["evaluation_additional_source_sha256"] == extra_sources
                    and value["candidate"] == name and value["declaration_digest"] == declaration["declaration_digest"],
                    "Cached native/stress source or candidate hashes differ")
        require(evaluation["calibration_identity"] == asset["calibration_identity"], "Native calibration identity differs")
        require(evaluation["dataset"] == "KolektorSDD2" and evaluation["model_kind"] == model_kind
                and evaluation["training_seed"] == 42 and evaluation["calibration_normal_count"] == 313
                and evaluation["preprocess"] == job["config"]["preprocess"]
                and evaluation["validation_best_epoch"] == summary["best_epoch"]
                and evaluation["validation_best_metric"] == summary["best_validation"]["pooled"]["image_pauc"],
                "Cached checkpoint selection or preprocessing metadata differs")
        verify_metrics(evaluation, asset["threshold"], positives, negatives, native=True)
        original_predictions = verify_predictions(evaluation, image_records, asset["threshold"])
        require(tuple(stress["results"]) == CONDITIONS, "The five stress conditions differ")
        baseline_predictions = None
        for condition, metrics in stress["results"].items():
            verify_metrics(metrics, asset["threshold"], positives, negatives)
            predictions = verify_predictions(metrics, image_records, asset["threshold"])
            if condition == "original":
                baseline_predictions = predictions
                require(all(p["predicted_defective"] == original_predictions[key]["predicted_defective"]
                            for key, p in predictions.items()), "Native/stress original per-image decisions differ")
            flips = sum(p["predicted_defective"] != baseline_predictions[key]["predicted_defective"] for key, p in predictions.items())
            accepted_to_flagged = sum(p["predicted_defective"] and not baseline_predictions[key]["predicted_defective"]
                                      for key, p in predictions.items())
            require(metrics["decision_flips"] == flips and metrics["accepted_to_flagged"] == accepted_to_flagged
                    and metrics["flagged_to_accepted"] == flips - accepted_to_flagged, "Stress decision-change counts differ")
            close(metrics["mean_absolute_score_change"], sum(abs(p["score"] - baseline_predictions[key]["score"])
                  for key, p in predictions.items()) / len(predictions), "stress mean absolute score change")
        require(stress["results"]["original"]["confusion"] == evaluation["confusion"], "Native/stress original decisions disagree")
        require(latency["input"] == benchmark_input and set(latency["devices"]) == {"cpu", "mps"}, "Latency input/device identity differs")
        for device, measurement in latency["devices"].items():
            require(measurement["device"] == device and measurement["checkpoint_sha256"] == asset["checkpoint_sha256"]
                    and measurement["measurements"] == 32 and measurement["warmup_forwards"] == 3
                    and measurement["model_kind"] == model_kind and measurement["score_definition"] == score_definition,
                    "Latency method or score identity differs")
            require(all(finite(measurement[k]) and measurement[k] > 0 for k in ("single_image_median_ms", "single_image_p95_ms"))
                    and measurement["single_image_p95_ms"] >= measurement["single_image_median_ms"], "Invalid latency measurements")
        for path in (run / "summary.json", run / "calibration-summary.json", history_path, selected_path, checkpoint_path):
            artifact_hashes[str(path.relative_to(root))] = sha(path)
        gallery_dir = root / "docs/results/ksdd2-decision-galleries" / name
        gallery_index_path, attribution_path = gallery_dir / "index.json", gallery_dir / "ATTRIBUTION.md"
        gallery = read(gallery_index_path)
        require(gallery["candidate"] == name and gallery["checkpoint_sha256"] == asset["checkpoint_sha256"]
                and gallery["experimental_status"] == STATUS and gallery["evaluation_freeze_digest"] == freeze_id
                and gallery["display_range"] == [0, 1]
                and gallery["model_kind"] == model_kind and gallery["score_definition"] == score_definition,
                "Gallery model/score/freeze/display identity differs")
        expected_groups = {short: min(3, evaluation["confusion"][full]) for short, full in
                           (("FP", "false_positive"), ("FN", "false_negative"), ("TP", "true_positive"), ("TN", "true_negative"))}
        require(gallery["groups"] == expected_groups and len(gallery["items"]) == sum(expected_groups.values()),
                "Gallery must retain up to three available examples from every decision group")
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
        public_models.append({"candidate": name, "selected_for_exploratory_evaluation": name == "decision_head",
                              "asset": asset, "training_summary": summary, "training_history": history,
                              "calibration_summary": calibration, **records})
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

    evidence = portable({"schema_version": "ksdd2-decision-results-1", "experimental_status": STATUS,
        "training_supervision": "real defective and normal training images with pixel annotations; random initialization",
        "reported_selection": "decision_head was chosen by the validation gate before exploratory evaluation",
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
            "Validation uncertainty describes reused-validation partial image AUROC only; it is not independent confirmation or a pixel AP interval.",
            "The models select different training epochs. Classification inputs are detached, so localization differences cannot be attributed to classification gradients or a pure same-epoch head effect.",
            "The head's segmentation map localizes pixel responses, not a causal attribution of its separate image-classification score.",
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
    selected = next(m for m in models if m["candidate"] == "decision_head")
    original = selected["evaluation"]
    counts = original["confusion"]
    paired = evidence["paired_validation_uncertainty"]
    interval = paired["percentile_95_interval"]
    lines = ["# KolektorSDD2 learned image decision: exploratory results", "",
        f"**The validation-preselected decision head detects {counts['true_positive']}/110 original-test defects and falsely flags {counts['false_positive']}/894 normal images. Its original-test point target is {outcome(original)}.**", "",
        "This completed, exactly two-run comparison uses random initialization and real defect masks in the separate supervised surface task. Every reused-test result is **development-inspected / exploratory**. Neither training completion nor a passing validation gate establishes independent reliability. This report performs no model promotion; the published v0.1.0 registry, live demo and historical evidence remain unchanged.", "",
        "[Portable evidence with full-precision predictions and hashes](KSDD2_DECISION_RESULTS.json) · [Declared method](../KSDD2_DECISION_SCREEN.md) · [Frozen implementation](../../experiments/ksdd2_decision/README.md) · [Previous acquisition experiment](KSDD2_ROBUSTNESS_RESULTS.md) · [Historical v0.1.0](CONTROLLED_STUDY.md)", "",
        "## What the comparison changes", "",
        "Both fresh seed-42 runs use the same base-16 U-Net, audited partitions, balanced batch of eight, Adam at 0.0003, positive-weight-3 BCE plus Dice, and 640-high × 256-wide letterboxing. Both retain the acquisition policy: 50% unchanged images; otherwise native-image brightness U(0.75, 1.25), then JPEG integer quality 60–95. Padding is excluded from the relevant losses and score statistics.", "",
        "Control averages the strongest 1% of valid sigmoid segmentation responses. The candidate learns an image classifier from detached bottleneck features, detached segmentation logits and valid map statistics, with image-label BCE. It adds 25,747 parameters to the 488,705-parameter backbone (514,452 total). Its raw image logit can be negative; greater means more defective, not a calibrated defect probability. This compact implementation adapts the ViCoS mixed segmentation/decision idea and is not a reproduction of the published architecture or results.", "",
        "**Classification gradients cannot update the segmentation network.** Control selected epoch 5 and stopped at 20; the head selected epoch 25 and stopped at 40. Both had the same 100-epoch cap and 15 stale-training-epoch stopping rule, but realized compute and selected checkpoints differ. Any localization difference includes checkpoint-selection effects; it is not evidence that a detached head directly improved segmentation gradients or a controlled same-epoch head-only effect.", "",
        "The returned spatial map remains the segmentation model's pixel response. It is not a causal explanation of the separate image-classification decision.", "",
        "## Validation selection and frozen gate", "",
        "Checkpoint selection uses pooled standardized partial image AUROC over false-positive rates 0–0.1, with the earliest checkpoint retained on an exact tie. The same frozen bank contains four variants of each of 350 sources, not 1,400 independent images. Model-space validation pixel AP excludes padding.", "",
        "| Run | Completed epochs | Selected epoch | Pooled partial AUROC | Pooled full AUROC | Pooled pixel AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in models:
        summary = model["training_summary"]; v = summary["best_validation"]["pooled"]
        lines.append(f"| {model['candidate']} | {summary['completed_epochs']} | {summary['best_epoch']} | {v['image_pauc']:.6f} | {v['image_auroc']:.6f} | {v['pixel_ap']:.6f} |")
    lines += ["", f"The predeclared gate passed: partial-AUROC gain **{gate['pooled_pauc_gain']:.9f}**, a **{100 * gate['pauc_shortfall_reduction']:.3f}% reduction in the control's 1−R shortfall** (required ≥10%, with strictly positive gain). Pooled pixel AP changed by **{gate['pooled_pixel_ap_change']:+.9f}**. Every per-condition partial-AUROC and pixel-AP regression was within the allowed 0.01. No condition-specific winning epochs were mixed together.", "",
        "| Validation condition | Control partial AUROC | Head partial AUROC | Partial-AUROC change | Pixel-AP change |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for condition in VALIDATION_CONDITIONS:
        a, b = gate["control"]["conditions"][condition], gate["candidate"]["conditions"][condition]
        delta = gate["condition_changes"][condition]
        lines.append(f"| {condition} | {a['image_pauc']:.6f} | {b['image_pauc']:.6f} | {delta['image_pauc']:+.6f} | {delta['pixel_ap']:+.6f} |")
    crosses = interval[0] <= 0 <= interval[1]
    lines += ["", f"The paired grouped-bootstrap validation partial-AUROC difference is **{paired['difference']:+.6f}**, with 95% interval **[{interval[0]:+.6f}, {interval[1]:+.6f}]**. " + ("It crosses zero. " if crosses else "It excludes zero. ") + "This is descriptive uncertainty on reused validation sources, not new independent significance evidence, an interval for pixel AP or a guarantee at the calibrated operating threshold. The 1,000 bootstrap samples retain all four variants of each sampled source; the interval did not determine the gate.", "",
        "## Original-test decisions and localization", "",
        "Both selected checkpoint hashes and the passing gate were fixed before either calibration. Each model then received one linear q90 threshold from the same 313 separate clean normal calibration scores. No test or perturbation result changed either threshold; positive calibration images remained unused.", "",
        "| Run | TP / defects | FN | FP / normals | TN | Recall [95% Wilson] | Normal false alarms [95% Wilson] | Point target |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- |"]
    for model in models:
        m = model["evaluation"]; c = m["confusion"]; u = m["uncertainty"]
        lines.append(f"| {model['candidate']} | {c['true_positive']}/110 | {c['false_negative']} | {c['false_positive']}/894 | {c['true_negative']} | {percent(m['defect_recall'])} {ci(u['defect_recall_interval'])} | {percent(m['normal_false_alarm_rate'])} {ci(u['normal_false_alarm_rate_interval'])} | {outcome(m)} |")
    lines += ["", "| Run | Image AUROC [95% image bootstrap] | Image AP | Native pixel AP | Alert precision [95% Wilson] | Frozen threshold |",
              "| --- | --- | ---: | ---: | --- | ---: |"]
    for model in models:
        m = model["evaluation"]; u = m["uncertainty"]
        lines.append(f"| {model['candidate']} | {m['image_auroc']:.6f} {ci(u['image_auroc_interval'], False)} | {m['image_average_precision']:.6f} | {m['pixel_average_precision']:.6f} | {percent(m['precision'])} {ci(u['precision_interval'])} | {m['threshold']:.12g} |")
    lines += ["", "The target requires both recall ≥90% and normal false alarms ≤10% on the measured dataset. Passing point estimates do not establish population bounds. Rate intervals use Wilson scores and AUROC intervals use 1,000 stratified image-bootstrap samples; they do not correct prior test inspection, domain shift, unknown physical-product groups or training-seed variability. Alert precision depends on the dataset's prevalence.",
        "Native pixel AP restores maps to the original image dimensions and uses unchanged native masks. It differs from model-space validation AP and has no estimated confidence interval. Resizing can discard details that interpolation cannot restore. Raw negative classification logits are valid and are never clamped to the 0–1 segmentation-map range.", "",
        "## Native defect-size groups", "", "| Run / annotated area | Detected / defective | Missed | Recall [95% Wilson] |", "| --- | --- | ---: | --- |"]
    for model in models:
        for name in AREA_BINS:
            group = model["evaluation"]["defect_area_groups"]["groups"][name]
            lines.append(f"| {model['candidate']} / {name} | {group['detected']}/{group['images']} | {group['missed']} | {percent(group['recall'])} {ci(group['recall_interval'])} |")
    lines += ["", "The bins use original-mask area divided by original-image area. Small groups have wide uncertainty, and an empty group has no recall estimate. Every evaluated image, including each failure, remains in the portable predictions.", "",
              "## Paired stress tests with fixed thresholds", "",
              "| Run / condition | Detected / defects | False alarms / normals | Recall [95% Wilson] | False alarms [95% Wilson] | Image AP | Decisions changed | Point target |",
              "| --- | --- | --- | --- | --- | ---: | ---: | --- |"]
    failures = []
    for model in models:
        for condition in CONDITIONS:
            m = model["stress"]["results"][condition]; c = m["confusion"]; u = m["uncertainty"]
            lines.append(f"| {model['candidate']} / {condition} | {c['true_positive']}/110 | {c['false_positive']}/894 | {percent(m['defect_recall'])} {ci(u['defect_recall_interval'])} | {percent(m['normal_false_alarm_rate'])} {ci(u['normal_false_alarm_rate_interval'])} | {m['image_average_precision']:.6f} | {m['decision_flips']} | {outcome(m)} |")
            if not point_target(m): failures.append(f"{model['candidate']}/{condition}")
    lines += ["", ("Point targets remain unmet for: " + ", ".join(f"`{name}`" for name in failures) + ".") if failures else "Both models pass all five measured point targets. This still does not establish independent or population-level reliability.",
        "These five conditions reuse the same source images and are paired observations. The original condition's decisions match native evaluation. Stress analysis measures image decisions and rankings only; no stress pixel AP is claimed. The named blur, brightness and JPEG transformations do not establish reliability under arbitrary camera/product/acquisition changes.", "",
        "## Auditable examples and errors", ""]
    total = sum(len(m["gallery"]["metadata"]["items"]) for m in models)
    lines += [f"The galleries contain **{total} real evaluated examples**, taking up to three available images from each false-positive, false-negative, true-positive and true-negative group per model. Highest-scoring false alarms and lowest-scoring misses expose the strongest errors; correct decisions use evenly spaced score ranks. The selection is illustrative, not an unbiased sample or a substitute for the complete confusion counts.", ""]
    for model in models:
        name = model["candidate"]
        lines.append(f"- [{name} gallery index](ksdd2-decision-galleries/{name}/index.json) · [Attribution](ksdd2-decision-galleries/{name}/ATTRIBUTION.md)")
    lines += ["", "All maps share the fixed 0–1 segmentation-activation scale. Captions retain frozen batched image scores and decisions; maps use the same weights in single-image inference. A head image logit is a separate quantity and is not on that color scale. Very small activations can appear dark, and the displayed segmentation map is not a causal attribution of the learned classifier."]
    for group, label in (("FP", "False alarm"), ("FN", "Missed defect")):
        example = next((item for item in selected["gallery"]["metadata"]["items"] if item["group"] == group), None)
        if example:
            lines += ["", f"{label} from the decision-head model:", "", f"![{label}; real KolektorSDD2 {example['source']}](ksdd2-decision-galleries/decision_head/{example['file']})"]
        else:
            lines += ["", f"The selected head has no {group} example in this evaluated set; no illustrative error was invented."]
    lines += ["", "## Local CPU and MPS timings", "", "| Run | Device | Warm median (ms) | Warm p95 (ms) | Measurements / warmups |", "| --- | --- | ---: | ---: | --- |"]
    for model in models:
        for device in ("cpu", "mps"):
            t = model["latency"]["devices"][device]
            lines.append(f"| {model['candidate']} | {device} | {t['single_image_median_ms']:.3f} | {t['single_image_p95_ms']:.3f} | {t['measurements']} / {t['warmup_forwards']} |")
    lines += ["", "Measurements ran sequentially under the shared scheduler lock with the same checksum-pinned clean normal calibration image. Warm model-forward/scoring time includes the learned head when present; loading, decoding, preprocessing, transfers, rendering and hosting overhead are excluded. The lock does not guarantee external system idleness. These local device timings are not hosted response-time guarantees.", "",
        "## Provenance and interpretation limits", "", "| Identity | Digest |", "| --- | --- |",
        f"| Training declaration | `{evidence['training_declaration']['declaration_digest']}` |",
        f"| Evaluation freeze | `{evidence['evaluation_status']['freeze_digest']}` |",
        f"| Audited split | `{evidence['evaluation_freeze']['split_digest']}` |"]
    for model in models:
        lines.append(f"| {model['candidate']} calibrated checkpoint | `{model['asset']['checkpoint_sha256']}` |")
    lines += ["", "The portable JSON preserves full predictions, original artifact hashes, source and score identities, calibration scores, selected-epoch histories, grouped validation evidence and uncertainty. Absolute user paths are replaced with portable identifiers. Original hashes refer to unmodified local records, not normalized copies; the portable evidence has its own digest. This renderer opens no dataset image or mask, performs no inference and changes no threshold.",
        "There is one training seed per condition and different validation-selected epochs. The head adds parameters and supervised image computation; it does not turn this into a supervision-matched comparison with normal-only MVTec. Prior test inspection makes every new same-dataset outcome exploratory. A genuinely untouched same-category holdout with documented acquisition/product groups is needed for new independent reliability claims. Historical v0.1.0 evidence and deployed weights remain separate; this report does not promote either new model.", ""]
    return "\n".join(lines)


def publish(root, output="docs/results"):
    root = Path(root).resolve()
    evidence = verify(root)
    directory = inside(root, output)
    payloads = {"KSDD2_DECISION_RESULTS.json": json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
                "KSDD2_DECISION_RESULTS.md": render(evidence)}
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
