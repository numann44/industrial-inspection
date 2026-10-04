"""Guards for the isolated, fixed-model exploratory evaluation handoff."""
import copy
import pytest

from scripts.evaluate_ksdd2_robustness import CONDITIONS, rate_evidence, target_met, validate_cached, validate_latency


def test_cached_evidence_rejects_changed_model_threshold_split_or_independence_claim():
    asset = {"checkpoint_sha256": "frozen-model", "threshold": 2e-7}
    value = {**asset, "split_digest": "frozen-split", "experimental_status": "exploratory", "evaluation_freeze_digest": "freeze", "pixel_metric_space": "original"}
    assert validate_cached(value, asset, "frozen-split", "evaluation", "freeze") is value
    for key, wrong in (("checkpoint_sha256", "other-model"), ("threshold", 1e-7),
                       ("split_digest", "other-split"), ("evaluation_freeze_digest", "old-code"), ("experimental_status", "frozen-held-out")):
        with pytest.raises(ValueError, match="different model"):
            validate_cached({**value, key: wrong}, asset, "frozen-split", "evaluation", "freeze")
    with pytest.raises(ValueError, match="Original-resolution"):
        validate_cached({**value, "pixel_metric_space": "model"}, asset, "frozen-split", "evaluation", "freeze")


def test_cached_stress_must_keep_all_five_declared_conditions():
    asset = {"checkpoint_sha256": "frozen-model", "threshold": 2e-7}
    value = {**asset, "split_digest": "split", "experimental_status": "exploratory", "evaluation_freeze_digest": "freeze", "results": dict.fromkeys(CONDITIONS, {})}
    validate_cached(value, asset, "split", "stress", "freeze")
    incomplete = copy.deepcopy(value)
    del incomplete["results"]["jpeg_quality_60"]
    with pytest.raises(ValueError, match="condition set"):
        validate_cached(incomplete, asset, "split", "stress", "freeze")


def test_joint_target_and_uncertainty_are_distinct():
    metrics = {"confusion": {"true_positive": 9, "false_negative": 1, "false_positive": 1, "true_negative": 9},
               "defect_recall": .9, "normal_false_alarm_rate": .1}
    assert target_met(metrics)
    intervals = rate_evidence(metrics)
    assert intervals["defect_recall_interval"][0] < .9
    assert intervals["normal_false_alarm_rate_interval"][1] > .1
    assert not target_met({**metrics, "defect_recall": .89})
    assert not target_met({**metrics, "normal_false_alarm_rate": .11})


def test_latency_cache_requires_both_devices_frozen_input_and_measurement_method():
    asset = {"checkpoint_sha256": "model"}
    input_record = {"source": "train/normal.png", "image_sha256": "image"}
    good = {"evaluation_freeze_digest": "freeze", "input": input_record,
            "devices": {device: {**asset, "device": device, "measurements": 32, "warmup_forwards": 3}
                        for device in ("cpu", "mps")}}
    validate_latency(good, asset, "freeze", input_record)
    for bad in ({}, {**good, "devices": {}}, {**good, "input": {}}, {**good, "evaluation_freeze_digest": "other"}):
        with pytest.raises(ValueError):
            validate_latency(bad, asset, "freeze", input_record)
    bad = copy.deepcopy(good)
    bad["devices"]["mps"]["measurements"] = 1
    with pytest.raises(ValueError, match="method or model"):
        validate_latency(bad, asset, "freeze", input_record)
