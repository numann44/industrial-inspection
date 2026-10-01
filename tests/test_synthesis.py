import torch

from inspection.synthesis import synthesize


def test_corruptions_have_exact_support_and_reproducible_masks():
    image = torch.rand((3, 64, 64), generator=torch.Generator().manual_seed(2))
    for mode in ("scratch", "appearance", "texture"):
        first = synthesize(image, torch.Generator().manual_seed(7), mode=mode)
        second = synthesize(image, torch.Generator().manual_seed(7), mode=mode)
        corrupted, mask = first
        assert torch.equal(corrupted, second[0])
        assert torch.equal(mask, second[1])
        assert 0 < mask.sum() < mask.numel()
        assert torch.equal(corrupted[:, mask[0] == 0], image[:, mask[0] == 0])
        assert (corrupted[:, mask[0] == 1] - image[:, mask[0] == 1]).abs().sum() > 0
        assert 0 <= corrupted.min() <= corrupted.max() <= 1


def test_normal_samples_are_clean_and_have_no_mask():
    image = torch.rand(3, 32, 32)
    corrupted, mask = synthesize(image, torch.Generator().manual_seed(3), normal_probability=1)
    assert torch.equal(corrupted, image)
    assert mask.sum() == 0
