# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the optional leakage measurement path."""

from __future__ import annotations

import unittest

import stim

from data.leakage import sample_measurements


class TestLeakageSampling(unittest.TestCase):
    def setUp(self):
        self.circuit = stim.Circuit("H 0\nCX 0 1\nM 0 1")

    def test_zero_leakage_uses_binary_stim_samples(self):
        samples = sample_measurements(
            self.circuit, shots=8, leakage_probability=0.0
        )
        self.assertEqual(samples.shape, (8, 2))
        self.assertEqual(samples.dtype, bool)

    def test_positive_leakage_returns_binary_measurements(self):
        samples = sample_measurements(
            self.circuit, shots=8, leakage_probability=1e-4
        )
        self.assertEqual(samples.shape, (8, 2))
        self.assertEqual(samples.dtype, bool)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_measurements(self.circuit, shots=1, leakage_probability=0.5001)


if __name__ == "__main__":
    unittest.main()
