import torch
from types import SimpleNamespace

from tridi.core.trainer import Trainer
from tridi.model.denoising.transformer_uni_3 import TransformertUni3WayModel


def test_geometry_adapter_starts_at_baseline():
    model = TransformertUni3WayModel(
        dim_sbj=325, dim_obj=9, dim_contact=128, dim_cond=4,
        dim_hidden=32, dim_timestep_embed=16, dim_output=462,
        num_layers=2, nhead=4, use_geometry=True,
    ).eval()
    inputs = [torch.randn(2, size) for size in (325, 9, 128, 4)]
    times = [torch.zeros(2, dtype=torch.long)] * 3
    geometry = torch.randn(2, 32, 3)
    with torch.no_grad():
        output_with_geometry = model(*inputs, *times, geometry=geometry)
        model.use_geometry = False
        output_without_geometry = model(*inputs, *times)
    torch.testing.assert_close(output_with_geometry, output_without_geometry)


def test_empty_low_noise_batch_has_finite_loss():
    trainer = object.__new__(Trainer)
    trainer.cfg = SimpleNamespace(train=SimpleNamespace(
        loss_t_stamp_threshold=250,
        losses={"smpl_v2v": 30.0, "obj_v2v": 10.0, "sbj_contacts": 20.0, "denoise_1": 1.0},
    ))
    output = SimpleNamespace(timesteps_sbj=torch.tensor([900]), timesteps_obj=torch.tensor([900]))
    loss, log = trainer.compute_loss(None, output, {"denoise_1": torch.tensor(2.0)})
    assert float(loss) == 2.0
    assert all(value == value for value in log.values())


if __name__ == "__main__":
    test_geometry_adapter_starts_at_baseline()
    test_empty_low_noise_batch_has_finite_loss()
