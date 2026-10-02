import numpy as np
import pytest
from PIL import Image

from inspection.reporting import (defect_area_groups, image_uncertainty, select_gallery_cases,
                                  wilson_interval, write_failure_gallery)


def test_image_uncertainty_uses_image_counts_and_fixed_threshold():
    result = image_uncertainty([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9], threshold=0.8, bootstrap_samples=50)
    assert result["image_auroc_interval"] == [1.0, 1.0]
    assert result["defect_recall_interval"] == wilson_interval(1, 2)
    assert result["normal_false_alarm_rate_interval"] == wilson_interval(0, 2)
    assert result["normal_images"] == 2
    assert result["defective_images"] == 2
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_gallery_selects_worst_errors_and_spaced_correct_ranks():
    records = [{"image": str(index), "label": label} for index, label in enumerate([0, 0, 1, 1, 1, 1, 1, 0])]
    groups = select_gallery_cases(records, [0.8, 0.95, 0.2, 0.01, 0.51, 0.7, 0.99, 0.1], threshold=0.5, limit_per_group=2)
    assert groups["FP"] == [1, 0]
    assert groups["FN"] == [3, 2]
    assert groups["TP"] == [4, 6]
    assert groups["TN"] == [7]


def test_native_area_bins_and_ground_truth_gallery(tmp_path):
    records = []
    for index, label in enumerate([0, 1, 1, 0]):
        path = tmp_path / f"{index}.png"
        Image.fromarray(np.full((32, 32, 3), 100 + index, dtype=np.uint8)).save(path)
        mask_path = None
        if label:
            mask_path = tmp_path / f"{index}_mask.png"
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[:8, :8] = 255
            Image.fromarray(mask).save(mask_path)
        records.append({"image": str(path), "mask": str(mask_path) if mask_path else None,
                        "label": label, "category": "metal_nut", "defect": "scratch" if label else "good"})
    scores = [0.9, 0.1, 0.8, 0.2]
    area = defect_area_groups(records, scores, threshold=0.5)
    assert area["groups"]["large_at_least_5_percent"]["images"] == 2
    assert area["groups"]["large_at_least_5_percent"]["detected"] == 1
    result = write_failure_gallery(records, [np.full((32, 32), score) for score in scores],
                                    scores, 0.5, tmp_path / "gallery")
    assert result["groups"] == {"FP": 1, "FN": 1, "TP": 1, "TN": 1}
    assert all(not item["source"].startswith("/") for item in result["items"])
    with Image.open(tmp_path / "gallery" / result["items"][0]["file"]) as image:
        assert image.size == (96, 96)
