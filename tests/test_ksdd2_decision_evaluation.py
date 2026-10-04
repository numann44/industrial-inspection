"""Fixed scoring/calibration contracts cannot drift across exploratory caches."""
import copy

import pytest

from experiments.ksdd2_decision.model import identity
from scripts.evaluate_ksdd2_decision import CONDITIONS, validate_cached, validate_latency


def asset_and_cache():
    kind, score = identity("decision_head")
    asset = {"candidate": "decision_head", "checkpoint_sha256": "frozen-head", "threshold": -3.6}
    value = {**asset, "split_digest": "split", "experimental_status": "exploratory",
             "evaluation_freeze_digest": "freeze", "model_kind": kind, "score_definition": score,
             "pixel_metric_space": "original"}
    return asset, value


def test_head_cache_preserves_negative_logit_threshold_and_raw_score_identity():
    asset, value = asset_and_cache()
    assert validate_cached(value, asset, "split", "evaluation", "freeze") is value
    for key, incorrect in (("threshold", .0266), ("score_definition", identity("control")[1]),
                           ("model_kind", identity("control")[0]), ("experimental_status", "frozen-held-out"),
                           ("checkpoint_sha256", "other-head"), ("split_digest", "other-split"),
                           ("evaluation_freeze_digest", "changed-adapter")):
        with pytest.raises(ValueError, match="different model"):
            validate_cached({**value, key: incorrect}, asset, "split", "evaluation", "freeze")
    with pytest.raises(ValueError, match="Original-resolution"):
        validate_cached({**value, "pixel_metric_space": "model"}, asset, "split", "evaluation", "freeze")


def test_stress_never_changes_threshold_or_omits_a_condition():
    asset, value = asset_and_cache()
    value["results"] = {name: {"threshold": asset["threshold"]} for name in CONDITIONS}
    assert validate_cached(value, asset, "split", "stress", "freeze") is value
    broken = copy.deepcopy(value)
    broken["results"]["brightness_1.2"]["threshold"] = -2.
    with pytest.raises(ValueError, match="threshold changed"):
        validate_cached(broken, asset, "split", "stress", "freeze")
    broken = copy.deepcopy(value)
    del broken["results"]["jpeg_quality_60"]
    with pytest.raises(ValueError, match="condition set"):
        validate_cached(broken, asset, "split", "stress", "freeze")


def test_head_timing_requires_same_scoring_contract_on_both_devices():
    asset, cache = asset_and_cache()
    source = {"source": "train/example.png", "image_sha256": "frozen-image"}
    timing = {"evaluation_freeze_digest": "freeze", "input": source,
              "devices": {device: {"device": device, "checkpoint_sha256": asset["checkpoint_sha256"],
                                    "model_kind": cache["model_kind"], "score_definition": cache["score_definition"],
                                    "measurements": 32, "warmup_forwards": 3}
                          for device in ("cpu", "mps")}}
    assert validate_latency(timing, asset, "freeze", source) is timing
    bad = copy.deepcopy(timing)
    bad["devices"]["cpu"]["score_definition"] = identity("control")[1]
    with pytest.raises(ValueError, match="method or model"):
        validate_latency(bad, asset, "freeze", source)
    for key, wrong in (("input", {}), ("evaluation_freeze_digest", "old"), ("devices", {})):
        with pytest.raises(ValueError, match="provenance or device"):
            validate_latency({**timing, key: wrong}, asset, "freeze", source)
