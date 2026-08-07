import numpy as np
import torch

from evaluation.loss_aware_predecoder import residual_detectors_and_frame
from qec.surface_code.deltakit_loss import (
    DeltakitHeraldedErasureSampler,
    memory_measurements_to_v2_input,
)


def test_zero_correction_and_raw_residual_preserves_ordinary_detectors():
    for basis in ("X", "Z"):
        sampler = DeltakitHeraldedErasureSampler(
            distance=3,
            n_rounds=3,
            basis=basis,
            pauli_error_probability=0.1,
            seed=1,
        )
        batch = sampler.sample_with_decoder_data(8)
        train_x = memory_measurements_to_v2_input(
            batch.features.measurements,
            batch.features.heralded_erasures,
            distance=3,
            n_rounds=3,
            basis=basis,
            code_rotation="XV",
        )
        prediction = torch.stack(
            [torch.zeros_like(train_x[:, 0]), torch.zeros_like(train_x[:, 0]), train_x[:, 0], train_x[:, 1]],
            dim=1,
        )
        residual, frame = residual_detectors_and_frame(
            prediction,
            batch.detectors,
            sampler.original_detector_indices,
            basis=basis,
        )
        assert np.array_equal(
            residual[:, sampler.original_detector_indices],
            batch.detectors[:, sampler.original_detector_indices],
        )
        assert not frame.any()
