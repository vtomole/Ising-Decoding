#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.neural_decoder import (
    BluvsteinFeedForwardDecoder,
    load_neural_logical_decoder,
)


class TestNeuralLogicalDecoder(unittest.TestCase):

    def _write_checkpoint(self, path: Path, input_size: int = 5) -> None:
        model = BluvsteinFeedForwardDecoder(
            input_size,
            hidden_sizes=(4,),
            use_batch_norm=False,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "metadata": {
                    "input_size": input_size,
                    "hidden_sizes": (4,),
                    "use_batch_norm": False,
                    "threshold": 0.5,
                },
            },
            path,
        )

    def test_load_checkpoint_and_decode_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "neural_decoder.pt"
            self._write_checkpoint(checkpoint_path)

            decoder = load_neural_logical_decoder(
                checkpoint_path,
                input_size=5,
                device="cpu",
                batch_size=2,
            )

            dets = np.zeros((3, 5), dtype=np.uint8)
            predictions = decoder.decode_batch(dets)

            self.assertEqual(predictions.shape, (3, 1))
            self.assertEqual(predictions.dtype, np.uint8)
            self.assertTrue(np.isin(predictions, [0, 1]).all())

    def test_input_size_mismatch_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "neural_decoder.pt"
            self._write_checkpoint(checkpoint_path, input_size=5)

            with self.assertRaisesRegex(ValueError, "input_size mismatch"):
                load_neural_logical_decoder(
                    checkpoint_path,
                    input_size=6,
                    device="cpu",
                )

    def test_plain_state_dict_checkpoint_infers_architecture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "plain_state_dict.pt"
            model = BluvsteinFeedForwardDecoder(
                5,
                hidden_sizes=(4,),
                use_batch_norm=False,
            )
            torch.save(model.state_dict(), checkpoint_path)

            decoder = load_neural_logical_decoder(
                checkpoint_path,
                input_size=5,
                device="cpu",
            )

            predictions = decoder.decode_batch(np.ones((2, 5), dtype=np.uint8))
            self.assertEqual(predictions.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
