import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loss_aware_teacher import _adjacency_masks, local_erasure_teacher_targets


class TestLossAwareTeacher(unittest.TestCase):
    def setUp(self):
        self.train_x = torch.zeros(1, 5, 1, 3, 3)
        self.train_x[:, 2:4] = 1  # All check-grid positions visible for this unit test.
        self.train_x[0, 4, 0, 0, 0] = 1
        self.x_adj, self.z_adj = _adjacency_masks(3, "XV")

    def test_no_flags_leaves_syndrome_as_residual(self):
        self.train_x[:, 4] = 0
        self.train_x[0, 0, 0, 1, 1] = 1
        target = local_erasure_teacher_targets(self.train_x)
        self.assertFalse(target[:, :2].any())
        self.assertTrue(torch.equal(target[:, 2], self.train_x[:, 0]))
        self.assertTrue(torch.equal(target[:, 3], self.train_x[:, 1]))

    def test_complete_adjacent_pattern_gets_local_corrections(self):
        for grid_index in torch.nonzero(self.x_adj[0], as_tuple=False).flatten():
            self.train_x[0, 0, 0].flatten()[grid_index] = 1
        for grid_index in torch.nonzero(self.z_adj[0], as_tuple=False).flatten():
            self.train_x[0, 1, 0].flatten()[grid_index] = 1
        target = local_erasure_teacher_targets(self.train_x)
        self.assertEqual(target[0, 0, 0, 0, 0].item(), 1)  # Z correction
        self.assertEqual(target[0, 1, 0, 0, 0].item(), 1)  # X correction
        self.assertFalse(target[0, 2].any())
        self.assertFalse(target[0, 3].any())

    def test_incomplete_pattern_is_left_for_global_decoder(self):
        self.train_x[0, 4, 0, 0, 0] = 0
        self.train_x[0, 4, 0, 1, 1] = 1
        adjacent = torch.nonzero(self.x_adj[4], as_tuple=False).flatten()
        self.train_x[0, 0, 0].flatten()[adjacent[0]] = 1
        target = local_erasure_teacher_targets(self.train_x)
        self.assertEqual(target[0, 0, 0, 1, 1].item(), 0)
        self.assertTrue(torch.equal(target[:, 2], self.train_x[:, 0]))
