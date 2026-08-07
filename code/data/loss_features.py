# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feature construction for heralded-erasure-aware pre-decoders."""

import torch


def append_heralded_erasure_channel(
    train_x: torch.Tensor,
    heralded_erasures: torch.Tensor,
) -> torch.Tensor:
    """Append a per-round heralded-erasure map to a v1 input tensor.

    Args:
        train_x: The standard v1 tensor of shape ``(B, 4, T, D, D)``.
        heralded_erasures: A binary tensor of shape ``(B, T, D, D)``.  A one
            means that the corresponding data-qubit location was flagged by a
            leakage-detection unit in that round and has been converted to a
            heralded erasure.

    Returns:
        A v2 tensor of shape ``(B, 5, T, D, D)``.  Channel 4 is the supplied
        heralded-erasure map.

    This function deliberately does not synthesize an all-zero channel.  A v2
    training example must carry real herald information.
    """
    if train_x.ndim != 5 or train_x.shape[1] != 4:
        raise ValueError("train_x must have shape (B, 4, T, D, D).")
    if heralded_erasures.ndim != 4:
        raise ValueError("heralded_erasures must have shape (B, T, D, D).")
    if tuple(train_x.shape[0:1] + train_x.shape[2:]) != tuple(heralded_erasures.shape):
        raise ValueError(
            "heralded_erasures must match train_x in batch, time, and spatial dimensions."
        )

    erasure_channel = heralded_erasures.to(device=train_x.device, dtype=train_x.dtype).unsqueeze(1)
    return torch.cat([train_x, erasure_channel], dim=1).contiguous()


__all__ = ["append_heralded_erasure_channel"]
