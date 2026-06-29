# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neural logical decoders for residual detector syndromes.

The default implementation is inspired by the surface-code ML decoder described
in Bluvstein et al., "Architectural mechanisms of a universal fault-tolerant
quantum computer": a supervised binary classifier built from fully connected
layers, BatchNorm, and GELU activations. We expose logits rather than an
in-module sigmoid so training can use BCEWithLogitsLoss.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


DEFAULT_HIDDEN_SIZES = (1024, 512, 256)


class BluvsteinFeedForwardDecoder(nn.Module):
    """Feedforward binary logical decoder for detector-syndrome features."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        *,
        use_batch_norm: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_size) <= 0:
            raise ValueError(f"input_size must be positive, got {input_size!r}")

        layers: list[nn.Module] = []
        in_features = int(input_size)
        for hidden in hidden_sizes:
            hidden = int(hidden)
            layers.append(nn.Linear(in_features, hidden))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.GELU())
            if float(dropout_p) > 0.0:
                layers.append(nn.Dropout(float(dropout_p)))
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).squeeze(-1)


class NeuralLogicalDecoder:
    """Small inference wrapper returning binary logical-observable predictions."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str,
        threshold: float = 0.5,
        batch_size: int = 8192,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.batch_size = max(int(batch_size), 1)

    @torch.inference_mode()
    def predict_proba_numpy(self, detectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(detectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        chunks: list[np.ndarray] = []
        for start in range(0, arr.shape[0], self.batch_size):
            batch = torch.from_numpy(arr[start:start + self.batch_size]).to(self.device)
            probs = torch.sigmoid(self.model(batch)).detach().cpu().numpy()
            chunks.append(probs.reshape(-1))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)

    @torch.inference_mode()
    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        probs = self.predict_proba_numpy(detectors)
        return (probs >= self.threshold).astype(np.uint8).reshape(-1, 1)


def _parse_hidden_sizes(value) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_HIDDEN_SIZES
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception as exc:
            raise ValueError(f"Invalid hidden_sizes string: {value!r}") from exc
        value = parsed
    if not isinstance(value, Iterable):
        raise ValueError(f"hidden_sizes must be iterable, got {type(value).__name__}")
    sizes = tuple(int(v) for v in value)
    if not sizes or any(v <= 0 for v in sizes):
        raise ValueError(f"hidden_sizes must be positive, got {sizes!r}")
    return sizes


def _load_raw_checkpoint(path: Path, device: torch.device | str) -> tuple[dict, dict]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            from safetensors import safe_open
        except Exception as exc:
            raise ImportError(
                "safetensors is required to load .safetensors decoder checkpoints"
            ) from exc
        state_dict = load_file(str(path), device=str(device))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
        return state_dict, metadata

    raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected neural decoder checkpoint type: {type(raw).__name__}")
    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    elif "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw
    metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata", {}), dict) else {}
    for key in ("input_size", "hidden_sizes", "threshold", "use_batch_norm", "dropout_p"):
        if key in raw and key not in metadata:
            metadata[key] = raw[key]
    return state_dict, metadata


def _infer_architecture_from_state_dict(
    state_dict: dict
) -> tuple[int | None, tuple[int, ...] | None]:
    linear_layers: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        if not key.startswith("net.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) != 3:
            continue
        try:
            layer_index = int(parts[1])
        except ValueError:
            continue
        if torch.is_tensor(value) and value.ndim == 2:
            linear_layers.append((layer_index, value))

    if len(linear_layers) < 1:
        return None, None
    linear_layers.sort(key=lambda item: item[0])
    input_size = int(linear_layers[0][1].shape[1])
    hidden_sizes = tuple(int(weight.shape[0]) for _, weight in linear_layers[:-1])
    return input_size, hidden_sizes or None


def load_neural_logical_decoder(
    checkpoint_path: str | os.PathLike[str],
    *,
    input_size: int,
    device: torch.device | str,
    threshold: float = 0.5,
    batch_size: int = 8192,
) -> NeuralLogicalDecoder:
    """Load a feedforward neural logical decoder checkpoint."""
    path = Path(os.path.expandvars(os.path.expanduser(str(checkpoint_path))))
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Neural decoder checkpoint not found: {path}")

    state_dict, metadata = _load_raw_checkpoint(path, device)
    clean_state = {
        (k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    inferred_input_size, inferred_hidden_sizes = _infer_architecture_from_state_dict(clean_state)
    ckpt_input_size = int(metadata.get("input_size", inferred_input_size or input_size))
    if ckpt_input_size != int(input_size):
        raise ValueError(
            f"Neural decoder input_size mismatch: checkpoint has {ckpt_input_size}, "
            f"current detector count is {input_size}"
        )

    hidden_sizes = _parse_hidden_sizes(metadata.get("hidden_sizes", inferred_hidden_sizes))
    inferred_batch_norm = any(
        k.startswith("net.") and k.endswith(".running_mean") for k in clean_state
    )
    use_batch_norm = str(metadata.get("use_batch_norm", inferred_batch_norm)).lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    dropout_p = float(metadata.get("dropout_p", 0.0))
    if "threshold" in metadata:
        threshold = float(metadata["threshold"])

    model = BluvsteinFeedForwardDecoder(
        ckpt_input_size,
        hidden_sizes,
        use_batch_norm=use_batch_norm,
        dropout_p=dropout_p,
    )
    model.load_state_dict(clean_state)
    return NeuralLogicalDecoder(model, device=device, threshold=threshold, batch_size=batch_size)


def resolve_neural_decoder_checkpoint(cfg) -> str:
    """Return the neural decoder checkpoint path from env/config, or empty string."""
    env_path = os.environ.get("PREDECODER_NEURAL_DECODER_CHECKPOINT", "").strip()
    if env_path:
        return env_path
    test_cfg = getattr(cfg, "test", None)
    if test_cfg is None:
        return ""
    return str(getattr(test_cfg, "neural_decoder_checkpoint", "") or "").strip()
