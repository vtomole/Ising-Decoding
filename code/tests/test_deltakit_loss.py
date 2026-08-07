import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import deltakit_stim  # noqa: F401
except ImportError:
    deltakit_stim = None

from qec.surface_code.deltakit_loss import DeltakitHeraldedErasureSampler


@unittest.skipIf(deltakit_stim is None, "deltakit-stim is not installed")
class TestDeltakitHeraldedErasureSampler(unittest.TestCase):
    def test_zero_leakage_has_zero_herald_map(self):
        sampler = DeltakitHeraldedErasureSampler(
            distance=3, n_rounds=3, leakage_probability=0.0, seed=1
        )
        batch = sampler.sample(4)
        self.assertEqual(batch.measurements.shape, (4, 33))
        self.assertEqual(batch.heralded_erasures.shape, (4, 3, 3, 3))
        self.assertFalse(batch.heralded_erasures.any())
        self.assertEqual(sampler.herald_detector_indices.shape, (3, 9))
        # One ordinary surface-code detector per check/round plus one detector
        # for every data-qubit herald event.
        self.assertEqual(sampler.circuit.num_detectors, 24 + 3 * 3 * 3)
        self.assertEqual(sampler.sample_train_x(2).shape, (2, 5, 3, 3, 3))

    def test_herald_map_uses_data_qubit_coordinates(self):
        sampler = DeltakitHeraldedErasureSampler(
            distance=3,
            n_rounds=3,
            leakage_probability=1.0,
            leakage_qubits=[0],
            seed=1,
        )
        batch = sampler.sample(3)
        self.assertTrue(np.all(batch.heralded_erasures[:, :, 0, 0] == 1))
        self.assertFalse(np.any(batch.heralded_erasures[:, :, 0, 1:]))
        self.assertFalse(np.any(batch.heralded_erasures[:, :, 1:, :]))
        self.assertTrue(np.all(sampler.sample_train_x(3)[:, 4, :, 0, 0].numpy() == 1))

    def test_stochastic_leakage_is_injected_once_per_round(self):
        sampler = DeltakitHeraldedErasureSampler(
            distance=3, n_rounds=3, leakage_probability=0.01, seed=1
        )
        leakage_lines = [line for line in str(sampler.circuit).splitlines() if line.startswith("LEAKAGE")]
        self.assertEqual(len(leakage_lines), 3)


if __name__ == "__main__":
    unittest.main()
