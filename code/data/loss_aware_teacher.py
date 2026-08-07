# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A conservative local teacher for heralded-erasure predecoder training.

This is not a replacement for global matching.  It only applies a data-qubit
Pauli-frame correction when all *visible* adjacent checks of the relevant type
fire at a heralded-erasure location.  All other cases remain in the residual
syndrome for ``LossAwareMatching``.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
)
from qec.surface_code.memory_circuit import SurfaceCode


@lru_cache(maxsize=None)
def _adjacency_masks(distance: int, code_rotation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return data-qubit-to-X/Z-check-grid adjacency masks, shape ``(D², D²)``."""
    code = SurfaceCode(
        distance,
        first_bulk_syndrome_type=code_rotation[0],
        rotated_type=code_rotation[1],
    )
    num_data = distance * distance
    x_masks = torch.zeros((num_data, num_data), dtype=torch.bool)
    z_masks = torch.zeros_like(x_masks)
    x_grid_indices = compute_stabX_to_data_index_map(distance, code_rotation)
    z_grid_indices = compute_stabZ_to_data_index_map(distance, code_rotation)
    for check_index, grid_index in enumerate(x_grid_indices):
        x_masks[torch.as_tensor(code.hx[check_index], dtype=torch.bool), grid_index] = True
    for check_index, grid_index in enumerate(z_grid_indices):
        z_masks[torch.as_tensor(code.hz[check_index], dtype=torch.bool), grid_index] = True
    return x_masks, z_masks


def local_erasure_teacher_targets(train_x: torch.Tensor, *, code_rotation: str = "XV") -> torch.Tensor:
    """Return conservative v1-style targets from a five-channel v2 input.

    ``train_x`` has channels ``[X syndrome, Z syndrome, X presence,
    Z presence, heralded erasure]``.  The output is
    ``[Z correction, X correction, residual X syndrome, residual Z syndrome]``.
    """
    if train_x.ndim != 5 or train_x.shape[1] != 5:
        raise ValueError("train_x must have shape (B, 5, T, D, D).")
    batch, _, rounds, distance, width = train_x.shape
    if distance != width:
        raise ValueError("train_x must have square spatial dimensions.")
    rotation = str(code_rotation).upper()
    if rotation not in {"XV", "XH", "ZV", "ZH"}:
        raise ValueError("code_rotation must be one of XV, XH, ZV, ZH.")

    x_adj, z_adj = _adjacency_masks(distance, rotation)
    x_adj = x_adj.to(device=train_x.device)
    z_adj = z_adj.to(device=train_x.device)
    n_data = distance * distance
    x_syndrome = train_x[:, 0].reshape(batch, rounds, n_data).bool()
    z_syndrome = train_x[:, 1].reshape(batch, rounds, n_data).bool()
    x_present = train_x[:, 2].reshape(batch, rounds, n_data).bool()
    z_present = train_x[:, 3].reshape(batch, rounds, n_data).bool()
    flags = train_x[:, 4].reshape(batch, rounds, n_data).bool()

    def unambiguous_decisions(
        syndrome: torch.Tensor,
        presence: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        # Count visible/firing adjacent checks with matrix multiplication.
        # This deliberately avoids constructing a (B, T, Q, N) broadcasted
        # tensor, which is prohibitively large for tutorial-scale shards.
        adjacency_int = adjacency.T.to(dtype=torch.int16)
        required_count = presence.to(torch.int16) @ adjacency_int
        fired_required = (syndrome & presence).to(torch.int16) @ adjacency_int
        return flags & (required_count > 0) & (fired_required == required_count)

    # X checks diagnose Z errors; Z checks diagnose X errors.
    z_correction = unambiguous_decisions(x_syndrome, x_present, x_adj)
    x_correction = unambiguous_decisions(z_syndrome, z_present, z_adj)

    def residual(syndrome: torch.Tensor, correction: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        toggles = torch.remainder(correction.to(torch.int16) @ adjacency.to(torch.int16), 2).bool()
        return syndrome ^ toggles

    residual_x = residual(x_syndrome, z_correction, x_adj)
    residual_z = residual(z_syndrome, x_correction, z_adj)
    return torch.stack(
        [
            z_correction.reshape(batch, rounds, distance, distance),
            x_correction.reshape(batch, rounds, distance, distance),
            residual_x.reshape(batch, rounds, distance, distance),
            residual_z.reshape(batch, rounds, distance, distance),
        ],
        dim=1,
    ).to(dtype=train_x.dtype)


__all__ = ["local_erasure_teacher_targets"]
