import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import deltakit_stim  # noqa: F401
except ImportError:
    deltakit_stim = None

from evaluation.loss_aware_matching import LossAwareMatching
from qec.surface_code.deltakit_loss import DeltakitHeraldedErasureSampler


@unittest.skipIf(deltakit_stim is None, "deltakit-stim is not installed")
class TestLossAwareMatching(unittest.TestCase):
    def setUp(self):
        self.matcher = LossAwareMatching(
            distance=3, n_rounds=3, pauli_error_probability=0.001
        )

    def test_no_flags_uses_cached_baseline_graph(self):
        sampler = DeltakitHeraldedErasureSampler(
            distance=3, n_rounds=3, pauli_error_probability=0.001, seed=3
        )
        batch = sampler.sample_with_decoder_data(4)
        prediction = self.matcher.decode_batch(
            batch.detectors, batch.features.heralded_erasures
        )
        baseline = self.matcher.matching_for_key(())
        np.testing.assert_array_equal(prediction, baseline.decode_batch(batch.detectors))
        self.assertEqual(self.matcher.cached_pattern_count, 1)

    def test_flagged_shot_builds_and_reuses_conditioned_graph(self):
        sampler = DeltakitHeraldedErasureSampler(
            distance=3,
            n_rounds=3,
            pauli_error_probability=0.001,
            forced_leakage_events=[(0, 0)],
            seed=4,
        )
        batch = sampler.sample_with_decoder_data(3)
        self.assertTrue(np.all(batch.features.heralded_erasures[:, 0, 0, 0] == 1))
        prediction = self.matcher.decode_batch(
            batch.detectors, batch.features.heralded_erasures
        )
        self.assertEqual(prediction.shape, batch.observables.shape)
        self.assertEqual(self.matcher.cached_pattern_count, 1)
        self.matcher.decode_batch(batch.detectors, batch.features.heralded_erasures)
        self.assertEqual(self.matcher.cached_pattern_count, 1)

    def test_multiple_flags_get_a_distinct_cached_graph(self):
        sampler = DeltakitHeraldedErasureSampler(
            distance=3,
            n_rounds=3,
            pauli_error_probability=0.001,
            forced_leakage_events=[(0, 0), (1, 1)],
            seed=5,
        )
        batch = sampler.sample_with_decoder_data(2)
        self.matcher.decode_batch(batch.detectors, batch.features.heralded_erasures)
        multi_flag_graph = self.matcher.matching_for_key(((0, 0), (1, 1)))
        baseline_graph = self.matcher.matching_for_key(())
        self.assertEqual(self.matcher.cached_pattern_count, 2)
        self.assertNotEqual(multi_flag_graph.num_edges, baseline_graph.num_edges)


if __name__ == "__main__":
    unittest.main()
