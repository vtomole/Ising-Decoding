"""Translate v2 predecoder outputs into loss-aware PyMatching inputs."""

from __future__ import annotations

import numpy as np
import torch

from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
)


def residual_detectors_and_frame(
    prediction: torch.Tensor,
    baseline_detectors: np.ndarray,
    original_detector_indices: np.ndarray,
    *,
    basis: str = "X",
    code_rotation: str = "XV",
) -> tuple[np.ndarray, np.ndarray]:
    """Build residual detector rows and the local logical-frame contribution.

    ``prediction`` is the thresholded v2 output in the tutorial convention
    ``[Z correction, X correction, residual X, residual Z]``.  Deltakit
    herald-detector bits stay untouched; only the ordinary MemoryCircuit
    detector positions are replaced by the predecoder residual syndrome.
    """
    if prediction.ndim != 5 or prediction.shape[1] != 4:
        raise ValueError("prediction must have shape (B, 4, T, D, D).")
    batch, _, rounds, distance, width = prediction.shape
    if distance != width:
        raise ValueError("prediction must have square spatial dimensions.")
    if baseline_detectors.shape[0] != batch:
        raise ValueError("baseline_detectors and prediction must have the same batch size.")
    if len(original_detector_indices) == 0:
        raise ValueError("original_detector_indices must not be empty.")

    basis = basis.upper()
    rotation = code_rotation.upper()
    if basis not in {"X", "Z"}:
        raise ValueError("basis must be X or Z.")

    # Grid locations not occupied by a stabilizer are ignored by the maps.
    x_indices = torch.as_tensor(compute_stabX_to_data_index_map(distance, rotation), device=prediction.device)
    z_indices = torch.as_tensor(compute_stabZ_to_data_index_map(distance, rotation), device=prediction.device)
    residual_x = prediction[:, 2].reshape(batch, rounds, distance * distance)[:, :, x_indices]
    residual_z = prediction[:, 3].reshape(batch, rounds, distance * distance)[:, :, z_indices]

    # This is the original MemoryCircuit detector order used by the existing
    # predecoder pipeline: one initial same-basis slice, then both stabilizer
    # types for remaining rounds, then boundary detectors.
    residual = np.asarray(baseline_detectors, dtype=np.uint8).copy()
    # The input formatter intentionally masks the unavailable syndrome type at
    # the first and final time boundaries.  Keep those detector values from
    # the sampled circuit instead of overwriting them with model outputs.
    ordinary = original_detector_indices
    position = 0
    initial = residual_x[:, 0] if basis == "X" else residual_z[:, 0]
    residual[:, ordinary[position:position + initial.shape[1]]] = initial.cpu().numpy()
    position += initial.shape[1]
    for time_index in range(1, rounds):
        x_values = residual_x[:, time_index]
        x_is_visible = not (basis == "Z" and time_index == rounds - 1)
        if x_is_visible:
            residual[:, ordinary[position:position + x_values.shape[1]]] = x_values.cpu().numpy()
        position += x_values.shape[1]
        z_values = residual_z[:, time_index]
        z_is_visible = not (basis == "X" and time_index == rounds - 1)
        if z_is_visible:
            residual[:, ordinary[position:position + z_values.shape[1]]] = z_values.cpu().numpy()
        position += z_values.shape[1]

    # Boundary detectors are not produced by the local model, so their sampled
    # values already retained above form the tail of the ordinary layout.
    data_correction = prediction[:, 0] if basis == "X" else prediction[:, 1]
    flat_correction = data_correction.reshape(batch, rounds, distance * distance)
    logical_support = torch.zeros(distance * distance, dtype=torch.uint8, device=prediction.device)
    if rotation in {"XV", "ZH"}:
        logical_support[:distance] = 1
    else:
        logical_support[::distance] = 1
    local_frame = (flat_correction.to(torch.uint8) * logical_support).sum(dim=2).sum(dim=1).remainder(2)
    return residual, local_frame.cpu().numpy().astype(np.uint8, copy=False)


__all__ = ["residual_detectors_and_frame"]
