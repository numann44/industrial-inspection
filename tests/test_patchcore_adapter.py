import numpy as np
import pytest
import torch
from copy import deepcopy

from inspection.patchcore_baseline import ExactTorchNN, validate_reference_provenance


def test_exact_search_matches_direct_squared_distance():
    generator = np.random.default_rng(42)
    bank = generator.normal(size=(31, 12)).astype(np.float32)
    queries = generator.normal(size=(300, 12)).astype(np.float32)
    search = ExactTorchNN()
    search.fit(bank)
    distances, indices = search.run(3, queries)
    direct = ((queries[:, None] - bank[None]) ** 2).sum(2)
    expected_indices = np.argsort(direct, axis=1)[:, :3]
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(distances, np.take_along_axis(direct, expected_indices, axis=1), rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        search.run(32, queries)


def test_official_reference_restores_memory_without_pretrained_download():
    pytest.importorskip("torchvision")
    pytest.importorskip("timm")
    from inspection.patchcore_baseline import PatchCoreReference
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1)
        model = PatchCoreReference(image_size=32)
        image = torch.rand(1, 3, 32, 32)
        features = np.asarray(model.core.embed(model.normalized(image)))
        model.core.anomaly_scorer.fit([features])
        expected_map, expected_score = model.predict_batch(image)
        exported = model.export_state()
        restored = PatchCoreReference(image_size=32)
        restored.load_export_state(exported)
        maps, scores = restored.predict_batch(image)
        torch.testing.assert_close(maps, expected_map, rtol=0, atol=0)
        torch.testing.assert_close(scores, expected_score, rtol=0, atol=0)
        assert maps.shape == (1, 1, 32, 32)
    finally:
        torch.set_num_threads(before)


@pytest.mark.parametrize("field", ["source_files", "adapter_sha256", "vendor_source_sha256", "packages"])
def test_reference_rejects_source_or_environment_drift(field):
    start = {"source_files": {"adapter.py": "original"}, "adapter_sha256": "original",
             "vendor_source_sha256": {"common.py": "original"}, "packages": {"torch": "fixed"},
             "created_at_utc": "start"}
    end = deepcopy(start)
    end["created_at_utc"] = "end"
    validate_reference_provenance(start, end)  # Timestamps naturally differ.
    end[field] = "changed"
    with pytest.raises(ValueError, match="changed during fitting"):
        validate_reference_provenance(start, end)
