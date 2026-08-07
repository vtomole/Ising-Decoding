# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-herald-pattern PyMatching graphs for converted leakage erasures."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pymatching
import stim

from qec.surface_code.deltakit_loss import DeltakitHeraldedErasureSampler


class LossAwareMatching:
    """Cache a Deltakit-conditioned PyMatching graph for every herald pattern.

    A key is the set of ``(round, data_qubit)`` locations flagged in one shot.
    The key's circuit forces leakage at those locations, performs the same LDU
    reset, and supplies its conditional DEM to PyMatching.  This mirrors the
    paper's precomputed-loss-DEM idea while remaining exact for the tutorial's
    simplified, one-location-per-round leakage injection model.

    Graph construction is intentionally cached.  The first occurrence of a
    new herald pattern is expensive; subsequent shots with that pattern only
    call ``decode_batch``.  The tutorial simulator uses an end-of-round LDU
    erasure model, so a ``(round, data_qubit)`` flag exactly identifies the
    deterministic conditional event used to build its graph.  A gate-resolved
    leakage model would instead need Perrin *et al.*-style marginalisation.
    """

    def __init__(
        self,
        *,
        distance: int,
        n_rounds: int,
        basis: str = "X",
        code_rotation: str = "XV",
        pauli_error_probability: float = 0.0,
    ) -> None:
        self._sampler_kwargs = dict(
            distance=int(distance),
            n_rounds=int(n_rounds),
            basis=basis,
            code_rotation=code_rotation,
            pauli_error_probability=float(pauli_error_probability),
            leakage_probability=0.0,
        )
        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self._cache: dict[tuple[tuple[int, int], ...], pymatching.Matching] = {}

    def _key_from_flags(self, flags: np.ndarray) -> tuple[tuple[int, int], ...]:
        if tuple(flags.shape) != (self.n_rounds, self.distance, self.distance):
            raise ValueError("Each herald map must have shape (rounds, distance, distance).")
        active = np.argwhere(flags != 0)
        return tuple(sorted((int(round_index), int(row * self.distance + column))
                            for round_index, row, column in active))

    def matching_for_key(self, key: tuple[tuple[int, int], ...]) -> pymatching.Matching:
        if key not in self._cache:
            sampler = DeltakitHeraldedErasureSampler(
                **self._sampler_kwargs,
                forced_leakage_events=key,
            )
            deltakit_dem = sampler.circuit.detector_error_model(decompose_errors=True)
            matching_dem = stim.DetectorErrorModel(str(deltakit_dem))
            self._cache[key] = pymatching.Matching.from_detector_error_model(matching_dem)
        return self._cache[key]

    @property
    def cached_pattern_count(self) -> int:
        return len(self._cache)

    def decode_batch(self, detectors: np.ndarray, heralded_erasures: np.ndarray) -> np.ndarray:
        """Decode detector rows using the graph conditioned on each row's flags."""
        if detectors.ndim != 2:
            raise ValueError("detectors must have shape (shots, num_detectors).")
        expected_flags = (detectors.shape[0], self.n_rounds, self.distance, self.distance)
        if tuple(heralded_erasures.shape) != expected_flags:
            raise ValueError(f"heralded_erasures must have shape {expected_flags}.")

        groups: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
        for shot, flags in enumerate(heralded_erasures):
            groups[self._key_from_flags(flags)].append(shot)

        predictions = None
        for key, indices in groups.items():
            matching = self.matching_for_key(key)
            decoded = matching.decode_batch(detectors[indices])
            if predictions is None:
                predictions = np.zeros((detectors.shape[0], decoded.shape[1]), dtype=decoded.dtype)
            predictions[indices] = decoded
        return predictions if predictions is not None else np.empty((0, 0), dtype=np.uint8)


__all__ = ["LossAwareMatching"]
