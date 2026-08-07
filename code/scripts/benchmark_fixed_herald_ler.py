#!/usr/bin/env python3
"""Large-sample LER benchmark conditioned on one fixed heralded erasure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.loss_aware_teacher import local_erasure_teacher_targets
from evaluation.loss_aware_matching import LossAwareMatching
from evaluation.loss_aware_predecoder import residual_detectors_and_frame
from model.predecoder import PreDecoderModelMemory_v2
from qec.surface_code.deltakit_loss import (
    DeltakitHeraldedErasureSampler,
    memory_measurements_to_v2_input,
)
from scripts.train_loss_aware_teacher import make_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--distance", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--pauli-p", type=float, default=0.002)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--round", type=int, default=None, dest="round_index")
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--column", type=int, default=None)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--oracle-local-teacher", action="store_true")
    args = parser.parse_args()

    if args.shots <= 0 or args.batch_size <= 0:
        raise ValueError("shots and batch-size must be positive.")
    row = args.distance // 2 if args.row is None else args.row
    column = args.distance // 2 if args.column is None else args.column
    round_index = args.rounds // 2 if args.round_index is None else args.round_index
    if not (0 <= row < args.distance and 0 <= column < args.distance and 0 <= round_index < args.rounds):
        raise ValueError("fixed herald location is outside the circuit.")

    device = torch.device(args.device)
    model = None
    if not args.oracle_local_teacher:
        model = PreDecoderModelMemory_v2(make_config(args.distance, args.rounds)).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

    data_qubit = row * args.distance + column
    sampler = DeltakitHeraldedErasureSampler(
        distance=args.distance,
        n_rounds=args.rounds,
        pauli_error_probability=args.pauli_p,
        leakage_probability=0.0,
        forced_leakage_events=[(round_index, data_qubit)],
        seed=args.seed,
    )
    matcher = LossAwareMatching(
        distance=args.distance,
        n_rounds=args.rounds,
        pauli_error_probability=args.pauli_p,
    )
    global_errors = 0
    predecoded_errors = 0
    completed = 0
    while completed < args.shots:
        count = min(args.batch_size, args.shots - completed)
        batch = sampler.sample_with_decoder_data(count)
        train_x = memory_measurements_to_v2_input(
            batch.features.measurements,
            batch.features.heralded_erasures,
            distance=args.distance,
            n_rounds=args.rounds,
            basis=sampler.basis,
            code_rotation=sampler.code_rotation,
        )
        if args.oracle_local_teacher:
            prediction = local_erasure_teacher_targets(train_x, code_rotation=sampler.code_rotation).to(torch.uint8)
        else:
            with torch.no_grad():
                prediction = (torch.sigmoid(model(train_x.to(device))) >= args.threshold).to(torch.uint8)
        residual, frame = residual_detectors_and_frame(
            prediction,
            batch.detectors,
            sampler.original_detector_indices,
            basis=sampler.basis,
            code_rotation=sampler.code_rotation,
        )
        global_prediction = matcher.decode_batch(batch.detectors, batch.features.heralded_erasures)
        predecoded_prediction = matcher.decode_batch(residual, batch.features.heralded_erasures)
        predecoded_prediction ^= frame[:, None]
        global_errors += int(np.any(global_prediction != batch.observables, axis=1).sum())
        predecoded_errors += int(np.any(predecoded_prediction != batch.observables, axis=1).sum())
        completed += count

    print(f"shots: {args.shots}")
    print(f"fixed herald: round={round_index}, row={row}, column={column}")
    print(f"predecoder source: {'local-teacher oracle' if args.oracle_local_teacher else 'v2 checkpoint'}")
    print(f"loss-aware MWPM LER: {global_errors / args.shots:.6g} ({global_errors}/{args.shots})")
    print(f"v2 + loss-aware MWPM LER: {predecoded_errors / args.shots:.6g} ({predecoded_errors}/{args.shots})")
    print(f"cached flag patterns: {matcher.cached_pattern_count}")


if __name__ == "__main__":
    main()
