#!/usr/bin/env python3
"""Generate a fixed Deltakit dataset for the tutorial's erasure model.

The shard intentionally stores raw measurements, detector data, and herald maps
in addition to the five-channel input.  It does not claim to contain v2
supervision labels; those require a separately specified local teacher.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.loss_aware_circuit_teacher import circuit_frame_teacher_targets
from data.loss_aware_teacher import local_erasure_teacher_targets
from qec.surface_code.deltakit_loss import (
    DeltakitHeraldedErasureSampler,
    memory_measurements_to_v2_input,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1_000)
    parser.add_argument("--pauli-p", type=float, default=0.002)
    parser.add_argument("--erasure-p", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--teacher", choices=("heuristic", "circuit-frame"), default="heuristic")
    args = parser.parse_args()

    sampler = DeltakitHeraldedErasureSampler(
        distance=args.distance,
        n_rounds=args.rounds,
        pauli_error_probability=args.pauli_p,
        leakage_probability=args.erasure_p,
        seed=args.seed,
    )
    batch = sampler.sample_with_frames(args.shots) if args.teacher == "circuit-frame" else sampler.sample_with_decoder_data(args.shots)
    train_x = memory_measurements_to_v2_input(
        batch.features.measurements,
        batch.features.heralded_erasures,
        distance=args.distance,
        n_rounds=args.rounds,
        basis=sampler.basis,
        code_rotation=sampler.code_rotation,
    ).numpy()
    if args.teacher == "circuit-frame":
        teacher_y = circuit_frame_teacher_targets(
            torch.from_numpy(train_x),
            torch.from_numpy(batch.data_x_frames),
            torch.from_numpy(batch.data_z_frames),
            code_rotation=sampler.code_rotation,
        ).numpy()
    else:
        teacher_y = local_erasure_teacher_targets(
            torch.from_numpy(train_x), code_rotation=sampler.code_rotation
        ).numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        train_x=train_x,
        local_teacher_y=teacher_y,
        measurements=batch.features.measurements,
        heralded_erasures=batch.features.heralded_erasures,
        detectors=batch.detectors,
        observables=batch.observables,
        distance=np.asarray(args.distance),
        n_rounds=np.asarray(args.rounds),
        pauli_error_probability=np.asarray(args.pauli_p),
        erasure_probability=np.asarray(args.erasure_p),
        seed=np.asarray(args.seed),
        teacher_mode=np.asarray(args.teacher),
    )
    print(f"wrote {args.output}")
    print(f"train_x shape: {train_x.shape}")
    print(f"local_teacher_y shape: {teacher_y.shape}")
    print(f"heralded erasures: {int(batch.features.heralded_erasures.sum())}")


if __name__ == "__main__":
    main()
