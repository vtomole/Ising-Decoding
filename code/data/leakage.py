# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional leakage sampling for Stim memory circuits.

The public circuit-level noise model remains unchanged.  When explicitly
enabled for an inference run, this module inserts a leakage transition after
each two-qubit Clifford gate and converts the leaky readout labels back to the
binary measurement record expected by Stim's measurement-to-detector converter.
"""

from __future__ import annotations

import importlib

import numpy as np
import stim


_TWO_QUBIT_GATES = frozenset({"CX", "CZ", "SWAP", "ISWAP"})
_ANNOTATION_PREFIXES = (
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
)


def _validate_probability(probability: float) -> float:
    probability = float(probability)
    if not 0.0 <= probability <= 0.5:
        raise ValueError(
            "Leakage probability must be between 0 and 0.5 inclusive; "
            "two transitions each occur with this probability."
        )
    return probability


def _instrument_leakage(circuit: stim.Circuit, channel_index: int = 0) -> stim.Circuit:
    """Insert a leaky identity after each two-qubit Clifford instruction.

    Detector and coordinate annotations are not instructions understood by the
    leaky sampler, and do not affect the measurement record, so they are
    omitted from the sampled circuit.  The original circuit is retained by the
    caller for measurement-to-detector conversion.
    """
    lines: list[str] = []
    for line in str(circuit).splitlines():
        stripped = line.strip()
        if stripped.startswith(_ANNOTATION_PREFIXES):
            continue
        lines.append(line)
        if not stripped or stripped in {"{", "}", "TICK"}:
            continue

        instruction, _, targets = stripped.partition(" ")
        gate_name = instruction.partition("(")[0]
        if gate_name in _TWO_QUBIT_GATES and targets:
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}I[leaky<{channel_index}>] {targets}")

    return stim.Circuit("\n".join(lines))


def _leaky_sampler(circuit: stim.Circuit, probability: float):
    try:
        leaky = importlib.import_module("leaky")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Leakage inference requires the optional 'leaky' package. "
            "Install it with `python -m pip install leakysim` in the Ising-Decoding environment."
        ) from exc

    channel = leaky.LeakyPauliChannel(2)
    computational = leaky.LeakageStatus(2)
    leaked_first = leaky.LeakageStatus(status=[0, 1])
    leaked_second = leaky.LeakageStatus(status=[0, 2])
    channel.add_transition(computational, computational, "II", 1.0 - 2.0 * probability)
    channel.add_transition(computational, leaked_first, "II", probability)
    channel.add_transition(computational, leaked_second, "II", probability)
    return leaky.Simulator(num_qubits=circuit.num_qubits, leaky_channels=[channel])


def sample_measurements(
    circuit: stim.Circuit,
    *,
    shots: int,
    leakage_probability: float | None,
) -> np.ndarray:
    """Return binary measurements from a standard or leakage-aware simulation.

    ``None`` and ``0`` deliberately use Stim directly, keeping normal inference
    free of the optional dependency and byte-for-byte on the existing path.
    """
    if leakage_probability is None:
        return circuit.compile_sampler().sample(shots=shots)

    probability = _validate_probability(leakage_probability)
    if probability == 0.0:
        return circuit.compile_sampler().sample(shots=shots)

    sampler = _leaky_sampler(circuit, probability)
    raw = np.asarray(
        sampler.sample(
            _instrument_leakage(circuit),
            shots=shots,
            readout_strategy=importlib.import_module("leaky").ReadoutStrategy.RawLabel,
        )
    )
    if not np.isin(raw, (0, 1, 2, 3)).all():
        raise RuntimeError("Leaky sampler returned an unsupported readout label.")
    # RawLabel uses 0/2 for a binary zero and 1/3 for a binary one.
    return np.logical_or(raw == 1, raw == 3)
