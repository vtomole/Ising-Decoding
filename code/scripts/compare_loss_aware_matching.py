#!/usr/bin/env python3
"""Compare static and herald-aware PyMatching on the tutorial erasure model.

Example:
    PYTHONPATH=code python code/scripts/compare_loss_aware_matching.py --shots 10000
"""

from __future__ import annotations

import argparse
import numpy as np
import pymatching
import stim

from evaluation.loss_aware_matching import LossAwareMatching
from qec.surface_code.deltakit_loss import DeltakitHeraldedErasureSampler


def logical_error_rate(predictions: np.ndarray, observables: np.ndarray) -> float:
    return float(np.any(predictions != observables, axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1_000)
    parser.add_argument("--pauli-p", type=float, default=0.002)
    parser.add_argument("--erasure-p", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    sampler = DeltakitHeraldedErasureSampler(
        distance=args.distance,
        n_rounds=args.rounds,
        pauli_error_probability=args.pauli_p,
        leakage_probability=args.erasure_p,
        seed=args.seed,
    )
    batch = sampler.sample_with_decoder_data(args.shots)
    matcher = LossAwareMatching(
        distance=args.distance,
        n_rounds=args.rounds,
        pauli_error_probability=args.pauli_p,
    )

    # A naïve decoder includes the *average* erasure channel in one static
    # graph, but does not condition on this shot's herald pattern.  Remove the
    # herald-only detector bits before decoding its ordinary syndrome.
    naive_dem = stim.DetectorErrorModel(
        str(sampler.circuit.detector_error_model(decompose_errors=True))
    )
    naive_matcher = pymatching.Matching.from_detector_error_model(naive_dem)
    naive_detectors = batch.detectors.copy()
    naive_detectors[:, sampler.herald_detector_indices.ravel()] = 0
    naive_predictions = naive_matcher.decode_batch(naive_detectors)
    aware_predictions = matcher.decode_batch(
        batch.detectors,
        batch.features.heralded_erasures,
    )

    print(f"shots: {args.shots}")
    print(f"heralded erasures: {int(batch.features.heralded_erasures.sum())}")
    print(f"naive LER: {logical_error_rate(naive_predictions, batch.observables):.6g}")
    print(f"loss-aware LER: {logical_error_rate(aware_predictions, batch.observables):.6g}")
    print(f"cached flag patterns: {matcher.cached_pattern_count}")


if __name__ == "__main__":
    main()
