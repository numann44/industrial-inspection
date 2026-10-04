"""Explicit display policy; never changes frozen inference values or decisions."""
from inspection.engine import render_maps


def display_result(result):
    """Use the full sigmoid activation range for supervised segmentation previews.

    The original engine's threshold-relative rendering is preserved in its own
    result and metadata. A tiny image threshold is not a useful pixel color range
    for this model. Neither range is a pixel decision threshold or probability.
    """
    if result["summary"]["model_kind"] != "supervised_segmentation":
        return result
    summary = dict(result["summary"])
    summary["checkpoint_display_scale"] = dict(summary["display_scale"])
    summary["display_scale"] = {
        "minimum": 0.0,
        "maximum": 1.0,
        "definition": "fixed 0–1 sigmoid activation display; not a pixel decision or defect probability",
    }
    summary["rendering_policy"] = "supervised-full-activation-range-v1"
    heatmap, overlay = render_maps(result["image"], result["native_map"], result["raw_map"], 1.0)
    return {**result, "summary": summary, "heatmap": heatmap, "overlay": overlay}
