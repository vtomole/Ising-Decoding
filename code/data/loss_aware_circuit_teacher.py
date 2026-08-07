"""Circuit-frame supervision for the loss-aware 3D predecoder.

The labels follow the four-channel convention used by the Chamberland-style
predecoder: data-frame differences (Z then X), followed by timelike
measurement-history corrections (X then Z).  Unlike the tutorial's heuristic
teacher, these targets are derived from Deltakit ``FlipSimulator`` trajectories.
"""

from __future__ import annotations

import torch

from data.loss_aware_teacher import _adjacency_masks


def circuit_frame_teacher_targets(
    train_x: torch.Tensor,
    data_x_frames: torch.Tensor,
    data_z_frames: torch.Tensor,
    *,
    code_rotation: str = "XV",
) -> torch.Tensor:
    """Return circuit-derived spacelike and timelike correction labels.

    ``data_x_frames`` and ``data_z_frames`` are cumulative hidden Pauli-frame
    bits after each LDU boundary, with shape ``(B, T, D, D)``.  The timelike
    targets are obtained by solving the temporal relation
    ``detector = data_syndrome ^ correction[t] ^ correction[t - 1]``.
    """
    if train_x.ndim != 5 or train_x.shape[1] != 5:
        raise ValueError("train_x must have shape (B, 5, T, D, D).")
    batch, _, rounds, distance, width = train_x.shape
    expected = (batch, rounds, distance, width)
    if distance != width or tuple(data_x_frames.shape) != expected or tuple(data_z_frames.shape) != expected:
        raise ValueError("frame tensors must have shape (B, T, D, D) matching train_x.")

    x_frame = data_x_frames.to(device=train_x.device, dtype=torch.uint8)
    z_frame = data_z_frames.to(device=train_x.device, dtype=torch.uint8)
    zero_x = torch.zeros_like(x_frame[:, :1])
    zero_z = torch.zeros_like(z_frame[:, :1])
    x_diff = x_frame ^ torch.cat([zero_x, x_frame[:, :-1]], dim=1)
    z_diff = z_frame ^ torch.cat([zero_z, z_frame[:, :-1]], dim=1)

    n_data = distance * distance
    x_adj, z_adj = _adjacency_masks(distance, code_rotation)
    # adjacency[data, check-grid].T maps a data-frame difference to the
    # corresponding X/Z detector-event grid.
    x_adjoint = x_adj.T.to(device=train_x.device, dtype=torch.int16)
    z_adjoint = z_adj.T.to(device=train_x.device, dtype=torch.int16)
    induced_x = (z_diff.reshape(batch, rounds, n_data).to(torch.int16) @ x_adjoint).remainder(2)
    induced_z = (x_diff.reshape(batch, rounds, n_data).to(torch.int16) @ z_adjoint).remainder(2)
    raw_x = train_x[:, 0].reshape(batch, rounds, n_data).to(torch.uint8)
    raw_z = train_x[:, 1].reshape(batch, rounds, n_data).to(torch.uint8)
    valid_x = train_x[:, 2].reshape(batch, rounds, n_data).bool()
    valid_z = train_x[:, 3].reshape(batch, rounds, n_data).bool()

    def timelike_targets(raw: torch.Tensor, induced: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        correction = torch.zeros_like(raw, dtype=torch.uint8)
        previous = torch.zeros_like(raw[:, 0], dtype=torch.uint8)
        for time_index in range(rounds):
            current = raw[:, time_index] ^ induced[:, time_index].to(torch.uint8) ^ previous
            current &= valid[:, time_index].to(torch.uint8)
            correction[:, time_index] = current
            previous = current
        return correction

    correction_x = timelike_targets(raw_x, induced_x, valid_x)
    correction_z = timelike_targets(raw_z, induced_z, valid_z)
    return torch.stack(
        [
            z_diff,
            x_diff,
            correction_x.reshape(batch, rounds, distance, distance),
            correction_z.reshape(batch, rounds, distance, distance),
        ],
        dim=1,
    ).to(dtype=train_x.dtype)


__all__ = ["circuit_frame_teacher_targets"]
