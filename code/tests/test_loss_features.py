import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loss_features import append_heralded_erasure_channel


class TestLossFeatures(unittest.TestCase):
    def test_appends_fifth_channel(self):
        train_x = torch.zeros(2, 4, 3, 3, 3)
        heralds = torch.zeros(2, 3, 3, 3)
        heralds[1, 2, 0, 1] = 1
        result = append_heralded_erasure_channel(train_x, heralds)
        self.assertEqual(result.shape, (2, 5, 3, 3, 3))
        self.assertEqual(result[1, 4, 2, 0, 1].item(), 1)

    def test_rejects_non_v1_input(self):
        with self.assertRaisesRegex(ValueError, "B, 4, T, D, D"):
            append_heralded_erasure_channel(torch.zeros(1, 5, 2, 3, 3), torch.zeros(1, 2, 3, 3))


if __name__ == "__main__":
    unittest.main()
