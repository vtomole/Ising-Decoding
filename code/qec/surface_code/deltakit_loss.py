# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deltakit-Stim sampling for leakage converted to heralded erasure.

The ordinary ``MemoryCircuit`` record is retained verbatim for the existing
syndrome formatter.  This module adds a separate, per-data-qubit herald record
after every stabilizer round.  A leakage reset (``RL``) immediately follows
the herald, implementing the effective LDU model used by the tutorial:
leakage is detected, converted into an erasure flag, then removed before the
next round.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
import torch

from data.loss_features import append_heralded_erasure_channel
from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
)
from qec.surface_code.memory_circuit import MemoryCircuit


_REC_REFERENCE = re.compile(r"rec\[-(\d+)\]")
_MEASUREMENT_GATES = frozenset({"M", "MX", "MY", "MR", "MRX", "MRY"})


@dataclass(frozen=True)
class DeltakitBatch:
    """Samples needed to construct a loss-aware training example.

    ``measurements`` has the unmodified MemoryCircuit measurement layout.
    ``heralded_erasures`` has shape ``(shots, rounds, distance, distance)``.
    """

    measurements: np.ndarray
    heralded_erasures: np.ndarray


@dataclass(frozen=True)
class DeltakitDecoderBatch:
    """Raw features plus detector events and true observable flips for one shot batch."""

    features: DeltakitBatch
    detectors: np.ndarray
    observables: np.ndarray


class DeltakitHeraldedErasureSampler:
    """Sample a surface-code memory circuit with Deltakit LDU boundaries.

    ``leakage_probability`` is applied once per selected data qubit at the
    end of each stabilizer round. Each Z-basis reset measurement (``MRZ``;
    rendered as ``MR`` by Stim's canonical formatter) ends one stabilizer
    round in the current MemoryCircuit construction; the sampler injects the
    effective erasure event, records ``HERALD_LEAKAGE_EVENT`` for every data
    qubit, and then applies ``RL`` to clear detected leakage.

    This is an LDU-level effective model, rather than a gate-resolved leakage
    model: a herald identifies the exact ``(round, data-qubit)`` erasure used
    by the loss-aware matching wrapper.
    """

    def __init__(
        self,
        *,
        distance: int,
        n_rounds: int,
        basis: str = "X",
        code_rotation: str = "XV",
        pauli_error_probability: float = 0.0,
        leakage_probability: float = 0.0,
        leakage_qubits: Sequence[int] | None = None,
        forced_leakage_events: Sequence[tuple[int, int]] | None = None,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= pauli_error_probability <= 1.0:
            raise ValueError("pauli_error_probability must lie in [0, 1].")
        if not 0.0 <= leakage_probability <= 1.0:
            raise ValueError("leakage_probability must lie in [0, 1].")
        try:
            import deltakit_stim
        except ImportError as error:
            raise ImportError(
                "DeltakitHeraldedErasureSampler requires deltakit-stim. "
                "Install it in the active environment before using the loss-aware backend."
            ) from error

        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self.basis = str(basis).upper()
        self.code_rotation = str(code_rotation).upper()
        self.pauli_error_probability = float(pauli_error_probability)
        self.leakage_probability = float(leakage_probability)

        base = MemoryCircuit(
            distance=self.distance,
            idle_error=self.pauli_error_probability,
            sqgate_error=self.pauli_error_probability,
            tqgate_error=self.pauli_error_probability,
            spam_error=(2.0 / 3.0) * self.pauli_error_probability,
            n_rounds=self.n_rounds,
            basis=self.basis,
            code_rotation=self.code_rotation,
            add_boundary_detectors=True,
        )
        self.data_qubits = tuple(int(q) for q in base.code.data_qubits)
        selected = self.data_qubits if leakage_qubits is None else tuple(int(q) for q in leakage_qubits)
        if not set(selected).issubset(self.data_qubits):
            raise ValueError("leakage_qubits must be surface-code data qubits.")
        self.leakage_qubits = frozenset(selected)
        forced_events = frozenset((int(round_index), int(qubit)) for round_index, qubit in (
            forced_leakage_events or ()
        ))
        if any(
            round_index < 0 or round_index >= self.n_rounds or qubit not in self.data_qubits
            for round_index, qubit in forced_events
        ):
            raise ValueError("forced_leakage_events must contain (round, data_qubit) pairs in range.")
        self.forced_leakage_events = forced_events

        instrumented_text, original_indices, herald_indices, herald_detector_indices = self._instrument(
            base.stim_circuit,
            self.data_qubits,
        )
        self.circuit = deltakit_stim.Circuit(instrumented_text)
        self._original_indices = np.asarray(original_indices, dtype=np.intp)
        self._herald_indices = np.asarray(herald_indices, dtype=np.intp)
        self.herald_detector_indices = np.asarray(herald_detector_indices, dtype=np.intp)
        self._sampler = self.circuit.compile_sampler(seed=seed)

    def _instrument(self, circuit, data_qubits: Sequence[int]):
        """Add leakage/LDU records while preserving all original rec references."""
        # Flatten first so each MR corresponds to exactly one physical round.
        lines = str(circuit.flattened()).splitlines()
        original_count = 0
        new_count = 0
        original_to_new: list[int] = []
        herald_indices: list[list[int]] = []
        herald_detector_indices: list[list[int]] = []
        output: list[str] = []
        round_index = 0
        detector_count = 0

        def rewrite_rec_references(line: str) -> str:
            def replacement(match: re.Match[str]) -> str:
                original_absolute = original_count - int(match.group(1))
                new_absolute = original_to_new[original_absolute]
                return f"rec[-{new_count - new_absolute}]"

            return _REC_REFERENCE.sub(replacement, line)

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            gate = stripped.split(maxsplit=1)[0]
            output.append(rewrite_rec_references(stripped))

            if gate in _MEASUREMENT_GATES:
                targets = stripped.split()[1:]
                count = len(targets)
                original_to_new.extend(range(new_count, new_count + count))
                original_count += count
                new_count += count

            # Stim's canonical formatter renders MRZ as MR.  In MemoryCircuit,
            # this is the Z-check reset-measurement that completes each round.
            if gate == "MR":
                forced_data = [
                    q for q in data_qubits if (round_index, q) in self.forced_leakage_events
                ]
                if forced_data:
                    output.append("LEAKAGE(1) " + " ".join(map(str, forced_data)))
                stochastic_data = [q for q in self.leakage_qubits if q not in forced_data]
                if stochastic_data and self.leakage_probability > 0.0:
                    output.append(
                        f"LEAKAGE({self.leakage_probability}) "
                        + " ".join(map(str, sorted(stochastic_data)))
                    )
                start = new_count
                data_targets = " ".join(map(str, data_qubits))
                output.append(f"HERALD_LEAKAGE_EVENT {data_targets}")
                # Make every herald bit a detector.  Besides retaining the
                # raw-record indices below, this is what exposes Deltakit's
                # adaptive leakage metadata in the circuit DEM for the later
                # global-decoder wrapper.  The order matches data_qubits.
                for offset in range(len(data_qubits), 0, -1):
                    output.append(f"DETECTOR rec[-{offset}]")
                herald_detector_indices.append(
                    list(range(detector_count, detector_count + len(data_qubits)))
                )
                detector_count += len(data_qubits)
                output.append(f"RL {data_targets}")
                herald_indices.append(list(range(start, start + len(data_qubits))))
                new_count += len(data_qubits)
                round_index += 1

            if gate == "DETECTOR":
                detector_count += 1

        if original_count != circuit.num_measurements:
            raise RuntimeError("Failed to preserve the original measurement-record layout.")
        if len(herald_indices) != self.n_rounds:
            raise RuntimeError(
                f"Expected {self.n_rounds} stabilizer rounds but found {len(herald_indices)} MR rounds."
            )
        return "\n".join(output), original_to_new, herald_indices, herald_detector_indices

    def sample(self, shots: int) -> DeltakitBatch:
        """Sample original measurements and an LDU herald map for ``shots`` shots."""
        if shots < 0:
            raise ValueError("shots must be non-negative.")
        records = self._sampler.sample(shots=int(shots))
        measurements = records[:, self._original_indices]
        heralds = records[:, self._herald_indices]
        heralds = heralds.reshape(shots, self.n_rounds, self.distance, self.distance)
        return DeltakitBatch(
            measurements=measurements.astype(np.uint8, copy=False),
            heralded_erasures=heralds.astype(np.uint8, copy=False),
        )

    def sample_train_x(self, shots: int) -> torch.Tensor:
        """Return v2's five-channel input tensor for fresh Deltakit samples."""
        batch = self.sample(shots)
        return memory_measurements_to_v2_input(
            batch.measurements,
            batch.heralded_erasures,
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis=self.basis,
            code_rotation=self.code_rotation,
        )

    def sample_with_decoder_data(self, shots: int) -> DeltakitDecoderBatch:
        """Sample features and matching-compatible detector/observable arrays together."""
        records = self._sampler.sample(shots=int(shots))
        measurements = records[:, self._original_indices]
        heralds = records[:, self._herald_indices].reshape(
            shots, self.n_rounds, self.distance, self.distance
        )
        converted = self.circuit.compile_m2d_converter().convert(
            measurements=records,
            append_observables=True,
        )
        detectors = converted[:, :self.circuit.num_detectors]
        observables = converted[:, self.circuit.num_detectors:]
        return DeltakitDecoderBatch(
            features=DeltakitBatch(
                measurements=measurements.astype(np.uint8, copy=False),
                heralded_erasures=heralds.astype(np.uint8, copy=False),
            ),
            detectors=detectors.astype(np.uint8, copy=False),
            observables=observables.astype(np.uint8, copy=False),
        )


def memory_measurements_to_v2_input(
    measurements: np.ndarray,
    heralded_erasures: np.ndarray,
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
) -> torch.Tensor:
    """Format preserved MemoryCircuit records and LDU flags as a v2 input.

    This follows the same four-channel convention as the Stim inference
    datapipe, then appends the real Deltakit herald map as channel 4.
    """
    d, rounds = int(distance), int(n_rounds)
    records = torch.as_tensor(measurements, dtype=torch.uint8)
    heralds = torch.as_tensor(heralded_erasures, dtype=torch.uint8)
    expected_records = rounds * (d * d - 1) + d * d
    if records.ndim != 2 or records.shape[1] != expected_records:
        raise ValueError(
            f"measurements must have shape (B, {expected_records}) for d={d}, rounds={rounds}."
        )
    if tuple(heralds.shape) != (records.shape[0], rounds, d, d):
        raise ValueError("heralded_erasures must have shape (B, rounds, distance, distance).")

    half = (d * d - 1) // 2
    frames = records[:, :-(d * d)].reshape(records.shape[0], rounds, d * d - 1)
    x_raw = frames[:, :, :half].permute(0, 2, 1).contiguous()
    z_raw = frames[:, :, half:].permute(0, 2, 1).contiguous()
    zeros = torch.zeros((records.shape[0], half, 1), dtype=torch.uint8)
    x_syn = torch.cat([zeros, x_raw], dim=2)
    z_syn = torch.cat([zeros, z_raw], dim=2)
    x_syn = (x_syn[:, :, 1:] ^ x_syn[:, :, :-1]).to(torch.float32)
    z_syn = (z_syn[:, :, 1:] ^ z_syn[:, :, :-1]).to(torch.float32)

    rotation = str(code_rotation).upper()
    x_map = torch.as_tensor(compute_stabX_to_data_index_map(d, rotation), dtype=torch.long)
    z_map = torch.as_tensor(compute_stabZ_to_data_index_map(d, rotation), dtype=torch.long)
    x_grid = torch.zeros(records.shape[0], d * d, rounds, dtype=torch.float32)
    z_grid = torch.zeros_like(x_grid)
    x_grid[:, x_map, :] = x_syn[:, :len(x_map), :]
    z_grid[:, z_map, :] = z_syn[:, :len(z_map), :]
    x_grid = x_grid.reshape(records.shape[0], d, d, rounds).permute(0, 3, 1, 2).contiguous()
    z_grid = z_grid.reshape(records.shape[0], d, d, rounds).permute(0, 3, 1, 2).contiguous()

    x_present = normalized_weight_mapping_Xstab_memory(d, rotation).reshape(d, d)
    z_present = normalized_weight_mapping_Zstab_memory(d, rotation).reshape(d, d)
    x_present = x_present.expand(records.shape[0], rounds, d, d).clone().float()
    z_present = z_present.expand(records.shape[0], rounds, d, d).clone().float()
    if str(basis).upper() == "X":
        z_grid[:, 0] = 0
        z_grid[:, -1] = 0
        z_present[:, 0] = 0
        z_present[:, -1] = 0
    elif str(basis).upper() == "Z":
        x_grid[:, 0] = 0
        x_grid[:, -1] = 0
        x_present[:, 0] = 0
        x_present[:, -1] = 0
    else:
        raise ValueError("basis must be 'X' or 'Z'.")

    v1_input = torch.stack([x_grid, z_grid, x_present, z_present], dim=1).contiguous()
    return append_heralded_erasure_channel(v1_input, heralds)


__all__ = [
    "DeltakitBatch",
    "DeltakitDecoderBatch",
    "DeltakitHeraldedErasureSampler",
    "memory_measurements_to_v2_input",
]
