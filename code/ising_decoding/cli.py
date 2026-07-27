# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Console entry point for the public Hydra workflow."""

from __future__ import annotations

import os
import sys


def _has_override(key: str) -> bool:
    return any(argument.lstrip("+").startswith(f"{key}=") for argument in sys.argv[1:])


def _append_environment_overrides() -> None:
    """Preserve the runner's convenient environment-variable interface."""
    mappings = (
        ("WORKFLOW", "workflow.task", False),
        ("EXPERIMENT_NAME", "exp_tag", True),
        ("DISTANCE", "distance", False),
        ("N_ROUNDS", "n_rounds", False),
        ("ISING_DECODING_MODEL_CHECKPOINT_FILE", "model_checkpoint_file", True),
    )
    for environment_name, config_key, add_key in mappings:
        value = os.environ.get(environment_name)
        if value and not _has_override(config_key):
            prefix = "+" if add_key else ""
            sys.argv.append(f"{prefix}{config_key}={value}")


def main() -> None:
    """Run the public workflow using the packaged configuration."""
    from workflows.run import run

    _append_environment_overrides()
    run()
