#!/usr/bin/env python3
"""Measure LER for loss-aware MWPM with and without the v2 predecoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation.loss_aware_matching import LossAwareMatching
from evaluation.loss_aware_predecoder import residual_detectors_and_frame
from model.predecoder import PreDecoderModelMemory_v2
from qec.surface_code.deltakit_loss import DeltakitHeraldedErasureSampler
from scripts.train_loss_aware_teacher import make_config


def logical_error_rate(predictions: np.ndarray, observables: np.ndarray) -> float:
    return float(np.any(predictions != observables, axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-shots", type=int, default=128)
    parser.add_argument("--basis", default="X")
    parser.add_argument("--code-rotation", default="XV")
    args = parser.parse_args()

    with np.load(args.dataset) as shard:
        count = min(int(args.max_shots), len(shard["train_x"]))
        train_x = torch.from_numpy(shard["train_x"][:count]).float()
        detectors = shard["detectors"][:count].astype(np.uint8, copy=False)
        heralds = shard["heralded_erasures"][:count].astype(np.uint8, copy=False)
        observables = shard["observables"][:count].astype(np.uint8, copy=False)
        distance = int(shard["distance"])
        rounds = int(shard["n_rounds"])
        pauli_p = float(shard["pauli_error_probability"])

    device = torch.device(args.device)
    model = PreDecoderModelMemory_v2(make_config(distance, rounds)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        prediction = (torch.sigmoid(model(train_x.to(device))) >= args.threshold).to(torch.uint8)

    # This zero-leakage instance supplies the invariant locations of ordinary
    # detector bits in the Deltakit-instrumented detector record.
    layout = DeltakitHeraldedErasureSampler(
        distance=distance,
        n_rounds=rounds,
        basis=args.basis,
        code_rotation=args.code_rotation,
        pauli_error_probability=pauli_p,
        leakage_probability=0.0,
    )
    residual_detectors, local_frame = residual_detectors_and_frame(
        prediction,
        detectors,
        layout.original_detector_indices,
        basis=args.basis,
        code_rotation=args.code_rotation,
    )
    matcher = LossAwareMatching(
        distance=distance,
        n_rounds=rounds,
        basis=args.basis,
        code_rotation=args.code_rotation,
        pauli_error_probability=pauli_p,
    )
    baseline_prediction = matcher.decode_batch(detectors, heralds)
    predecoded_prediction = matcher.decode_batch(residual_detectors, heralds)
    predecoded_prediction ^= local_frame[:, None]

    print(f"shots: {count}")
    print(f"heralded erasures: {int(heralds.sum())}")
    print(f"loss-aware MWPM LER: {logical_error_rate(baseline_prediction, observables):.6g}")
    print(f"v2 + loss-aware MWPM LER: {logical_error_rate(predecoded_prediction, observables):.6g}")
    print(f"cached flag patterns: {matcher.cached_pattern_count}")


if __name__ == "__main__":
    main()
