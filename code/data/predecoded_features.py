"""Residual features for a frozen Chamberland predecoder."""

from __future__ import annotations

import torch

from data.loss_aware_teacher import _adjacency_masks


def chamberland_residual_and_frame(
    train_x4: torch.Tensor, prediction: torch.Tensor, *, code_rotation: str = "XV", basis: str = "X"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply four Chamberland output channels to a four-channel input.

    Returns residual X/Z detector grids and the local logical-frame bit.
    """
    b, channels, rounds, distance, width = train_x4.shape
    if channels != 4 or prediction.shape != (b, 4, rounds, distance, width) or distance != width:
        raise ValueError("Expected matching (B, 4, T, D, D) input and prediction tensors.")
    n = distance * distance
    x_adj, z_adj = _adjacency_masks(distance, code_rotation)
    # CUDA does not implement matrix multiplication for int16. The sums are
    # at most four, so float32 parity arithmetic is exact here.
    z_data = prediction[:, 0].reshape(b, rounds, n).to(torch.float32)
    x_data = prediction[:, 1].reshape(b, rounds, n).to(torch.float32)
    induced_x = (z_data @ x_adj.T.to(prediction.device, torch.float32)).remainder(2).to(torch.uint8)
    induced_z = (x_data @ z_adj.T.to(prediction.device, torch.float32)).remainder(2).to(torch.uint8)
    tx = prediction[:, 2].reshape(b, rounds, n).to(torch.uint8)
    tz = prediction[:, 3].reshape(b, rounds, n).to(torch.uint8)
    previous_x = torch.cat([torch.zeros_like(tx[:, :1]), tx[:, :-1]], dim=1)
    previous_z = torch.cat([torch.zeros_like(tz[:, :1]), tz[:, :-1]], dim=1)
    residual = torch.stack([
        (train_x4[:, 0].reshape(b, rounds, n).to(torch.uint8) ^ induced_x ^ tx ^ previous_x).reshape(b, rounds, distance, distance),
        (train_x4[:, 1].reshape(b, rounds, n).to(torch.uint8) ^ induced_z ^ tz ^ previous_z).reshape(b, rounds, distance, distance),
    ], dim=1)
    support = torch.zeros(n, dtype=torch.uint8, device=prediction.device)
    if code_rotation.upper() in {"XV", "ZH"}:
        support[:distance] = 1
    else:
        support[::distance] = 1
    correction = prediction[:, 0] if basis.upper() == "X" else prediction[:, 1]
    frame = (correction.reshape(b, rounds, n).to(torch.uint8) * support).sum((1, 2)).remainder(2)
    return residual, frame
