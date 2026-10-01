import torch

from inspection.losses import inspection_loss
from inspection.model import InspectionModel, anomaly_score
from inspection.synthesis import synthesize


def test_anomaly_score_preserves_local_defect_signal():
    normal = torch.full((1, 1, 32, 32), -8.0)
    defect = normal.clone()
    defect[:, :, 10:14, 10:14] = 8.0
    assert anomaly_score(defect).item() > 0.99
    assert anomaly_score(normal).item() < 0.01


def test_random_initialized_model_can_learn_fixed_synthetic_batch():
    # A tiny overfit check catches broken gradients, targets and loss wiring.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        clean = torch.rand((3, 32, 32))
        image, mask = synthesize(clean, torch.Generator().manual_seed(9), mode="appearance")
        clean, image, mask = clean[None], image[None], mask[None]
        model = InspectionModel(base_channels=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        initial = inspection_loss(*model(image), clean, mask)[0].item()
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = inspection_loss(*model(image), clean, mask)
            loss.backward()
            optimizer.step()
        final = inspection_loss(*model(image), clean, mask)[0].item()
        assert final < initial * 0.75, (initial, final)
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    finally:
        torch.set_num_threads(previous_threads)
