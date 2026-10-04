import copy
import numpy as np
from PIL import Image

from demo_visuals import display_result


def test_preview_exposes_strong_region_without_changing_frozen_prediction():
    raw = np.array([[1e-5, 1e-7], [.9, .4]], dtype=np.float32)
    summary = {"model_kind": "supervised_segmentation", "score": .9,
               "threshold": 2.276815479262951e-7, "decision": "defective",
               "checkpoint_sha256": "immutable-checkpoint",
               "display_scale": {"minimum": 0., "maximum": 4.553630958525902e-7}}
    original = copy.deepcopy(summary)
    result = {"summary": summary, "image": Image.new("RGB", (2, 2), (64, 64, 64)),
              "raw_map": raw, "native_map": raw, "heatmap": None, "overlay": None}
    shown = display_result(result)
    assert summary == original
    for key in ("score", "threshold", "decision", "checkpoint_sha256"):
        assert shown["summary"][key] == summary[key]
    assert shown["raw_map"] is raw and shown["native_map"] is raw
    assert shown["summary"]["checkpoint_display_scale"] == summary["display_scale"]
    assert shown["summary"]["display_scale"]["maximum"] == 1.
    heatmap = np.asarray(shown["heatmap"])
    assert heatmap[0].max() == 0 and heatmap[1, 0] > heatmap[1, 1] > 0
    assert np.array_equal(np.asarray(shown["overlay"])[0], np.asarray(result["image"])[0])
    # Display policy remains fixed across images, never stretched per image.
    second = display_result({**result, "raw_map": raw / 2, "native_map": raw / 2})
    assert second["summary"]["display_scale"] == shown["summary"]["display_scale"]


def test_existing_normal_only_rendering_contract_is_unchanged():
    result = {"summary": {"model_kind": "joint_reconstruction_segmentation"}}
    assert display_result(result) is result
