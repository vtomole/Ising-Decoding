# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate public configuration and model files in source and wheel installs."""

from __future__ import annotations

from pathlib import Path
import sysconfig


_MODEL_FILENAMES = {
    "fast": "Ising-Decoder-SurfaceCode-1-Fast.pt",
    "accurate": "Ising-Decoder-SurfaceCode-1-Accurate.pt",
}


def _source_root() -> Path:
    """Return the repository root when running from a source checkout."""
    return Path(__file__).resolve().parents[2]


def _installed_data_root() -> Path:
    return Path(sysconfig.get_path("data")) / "ising_decoding"


def _data_root() -> Path:
    """Prefer installed package data, with a source-checkout fallback."""
    installed = _installed_data_root()
    if (installed / "conf" / "config_public.yaml").is_file():
        return installed
    return _source_root()


def public_config_dir() -> Path:
    """Return the directory containing the public Hydra configuration."""
    return _data_root() / "conf"


def model_path(model: str = "fast") -> Path:
    """Return the path to a bundled pretrained model.

    The model name is either ``fast`` (R=9) or ``accurate`` (R=13).  The
    package includes these Git-LFS-managed files as install data, so a regular
    ``pip install`` does not need a repository path at runtime.
    """
    try:
        filename = _MODEL_FILENAMES[model.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(_MODEL_FILENAMES))
        raise ValueError(f"Unknown model {model!r}; choose one of: {choices}.") from exc

    path = _data_root() / "models" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled {model} model is missing: {path}. Reinstall Ising-Decoding with Git LFS available."
        )
    if path.read_bytes()[:32].startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            "The bundled model is a Git LFS pointer, not model weights. Install git-lfs and reinstall the package."
        )
    return path
