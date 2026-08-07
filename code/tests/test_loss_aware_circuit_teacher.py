import torch

from data.loss_aware_circuit_teacher import circuit_frame_teacher_targets
from data.loss_aware_teacher import _adjacency_masks
from qec.surface_code.deltakit_loss import (
    DeltakitHeraldedErasureSampler,
    memory_measurements_to_v2_input,
)


def test_circuit_teacher_cancels_visible_detector_history():
    sampler = DeltakitHeraldedErasureSampler(
        distance=3, n_rounds=3, pauli_error_probability=0.02, leakage_probability=0.1, seed=3
    )
    batch = sampler.sample_with_frames(32)
    train_x = memory_measurements_to_v2_input(
        batch.features.measurements,
        batch.features.heralded_erasures,
        distance=3,
        n_rounds=3,
        basis=sampler.basis,
        code_rotation=sampler.code_rotation,
    )
    target = circuit_frame_teacher_targets(
        train_x,
        torch.from_numpy(batch.data_x_frames),
        torch.from_numpy(batch.data_z_frames),
        code_rotation=sampler.code_rotation,
    )
    b, _, t, d, _ = target.shape
    x_adj, z_adj = _adjacency_masks(d, sampler.code_rotation)
    z_diff = target[:, 0].reshape(b, t, d * d).to(torch.int16)
    x_diff = target[:, 1].reshape(b, t, d * d).to(torch.int16)
    induced_x = (z_diff @ x_adj.T.to(torch.int16)).remainder(2).to(torch.uint8)
    induced_z = (x_diff @ z_adj.T.to(torch.int16)).remainder(2).to(torch.uint8)
    corr_x = target[:, 2].reshape(b, t, d * d).to(torch.uint8)
    corr_z = target[:, 3].reshape(b, t, d * d).to(torch.uint8)
    prev_x = torch.cat([torch.zeros_like(corr_x[:, :1]), corr_x[:, :-1]], dim=1)
    prev_z = torch.cat([torch.zeros_like(corr_z[:, :1]), corr_z[:, :-1]], dim=1)
    residual_x = train_x[:, 0].reshape(b, t, d * d).to(torch.uint8) ^ induced_x ^ corr_x ^ prev_x
    residual_z = train_x[:, 1].reshape(b, t, d * d).to(torch.uint8) ^ induced_z ^ corr_z ^ prev_z
    assert not (residual_x & train_x[:, 2].reshape(b, t, d * d).to(torch.uint8)).any()
    assert not (residual_z & train_x[:, 3].reshape(b, t, d * d).to(torch.uint8)).any()
