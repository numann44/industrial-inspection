import numpy as np
import pytest
from PIL import Image

from inspection.evaluate import compute_metrics, original_pixel_metrics


def test_frozen_threshold_and_ranking_are_different():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.7, 0.9]
    masks = np.zeros((4, 1, 2, 2), dtype=np.uint8)
    masks[2:, :, 0, 0] = 1
    maps = np.zeros_like(masks, dtype=np.float32)
    maps[masks.astype(bool)] = 0.9
    result = compute_metrics(labels, scores, masks, maps, threshold=0.8)
    assert result["image_auroc"] == 1.0
    assert result["pixel_average_precision"] == 1.0
    assert result["confusion"] == {
        "true_positive": 1, "false_positive": 0, "true_negative": 2, "false_negative": 1}
    assert result["defect_recall"] == 0.5
    assert result["threshold"] == 0.8


def test_absent_classes_produce_explicit_null_metrics():
    result = compute_metrics([0, 0], [0.1, 0.8], np.zeros((2, 1, 2, 2)),
                             np.zeros((2, 1, 2, 2)), threshold=0.5)
    assert result["image_auroc"] is None
    assert result["pixel_average_precision"] is None
    assert result["defect_recall"] is None
    assert result["normal_false_alarm_rate"] == 0.5


def test_nonfinite_predictions_and_misaligned_masks_rejected():
    with pytest.raises(ValueError, match="finite"):
        compute_metrics([0, 1], [0.1, np.nan], np.zeros((2, 1, 2, 2)),
                        np.zeros((2, 1, 2, 2)), threshold=0.5)
    with pytest.raises(ValueError, match="aligned"):
        compute_metrics([0, 1], [0.1, 0.9], np.zeros((2, 1, 2, 2)),
                        np.zeros((2, 1, 4, 4)), threshold=0.5)


def test_original_pixel_space_preserves_tiny_defect(tmp_path):
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[0, 0] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)
    assert not np.asarray(Image.fromarray(mask).resize((128, 128), Image.Resampling.NEAREST)).any()
    maps = np.zeros((1, 1, 128, 128), dtype=np.float32)
    maps[0, 0, 0, 0] = 1
    result = original_pixel_metrics([{"category": "metal_nut", "mask": str(path),
                                       "original_size": [512, 512]}], maps)
    assert result["pixel_metric_space"] == "original"
    assert result["pixel_average_precision_by_category"]["metal_nut"]["positive_pixels"] == 1
    assert result["pixel_average_precision"] > 0
