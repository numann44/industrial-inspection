import torch

from inspection.synthesis import foreground_mask, synthesize


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


def test_foreground_synthesis_preserves_background_and_central_hole():
    image = torch.full((3, 64, 64), 0.08)
    generator = torch.Generator().manual_seed(12)
    image[:, 8:56, 8:56] = 0.45 + torch.rand((3, 48, 48), generator=generator) * 0.08
    image[:, 26:38, 26:38] = 0.08
    surface = foreground_mask(image)
    assert not surface[:, 26:38, 26:38].any()
    assert not surface[:, :7].any()
    assert surface[:, 12:24, 12:24].all()
    for mode in ("scratch", "lighting", "texture", "warp", "appearance"):
        first = synthesize(image, torch.Generator().manual_seed(8), mode=mode, strategy="foreground")
        second = synthesize(image, torch.Generator().manual_seed(8), mode=mode, strategy="foreground")
        corrupted, mask = first
        assert torch.equal(corrupted, second[0]) and torch.equal(mask, second[1])
        assert mask.sum() > 0
        assert not (mask.bool() & ~surface).any()
        assert torch.equal(corrupted[:, mask[0] == 0], image[:, mask[0] == 0])
        assert (corrupted[:, mask[0] == 1] - image[:, mask[0] == 1]).abs().sum() > 0
        assert 0 <= corrupted.min() <= corrupted.max() <= 1


def test_foreground_normal_probability_and_empty_surface_are_safe():
    image = torch.full((3, 32, 32), 0.1)
    for probability in (0, 1):
        corrupted, mask = synthesize(image, torch.Generator().manual_seed(3),
                                     normal_probability=probability, strategy="foreground")
        assert torch.equal(corrupted, image)
        assert not mask.any()
