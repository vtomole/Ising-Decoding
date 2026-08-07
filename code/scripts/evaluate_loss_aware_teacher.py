#!/usr/bin/env python3
"""Evaluate a v2 checkpoint against local-teacher labels in a fixed shard."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.train_loss_aware_teacher import make_config
from model.predecoder import PreDecoderModelMemory_v2


def binary_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    predicted_positive = prediction.bool()
    target_positive = target.bool()
    true_positive = (predicted_positive & target_positive).sum().item()
    false_positive = (predicted_positive & ~target_positive).sum().item()
    false_negative = (~predicted_positive & target_positive).sum().item()
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    accuracy = (predicted_positive == target_positive).float().mean().item()
    return precision, recall, accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device)
    with np.load(args.dataset) as shard:
        train_x = torch.from_numpy(shard["train_x"]).float()
        teacher_y = torch.from_numpy(shard["local_teacher_y"]).float()
        distance = int(shard["distance"])
        rounds = int(shard["n_rounds"])
    model = PreDecoderModelMemory_v2(make_config(distance, rounds)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(train_x.to(device)).cpu()
        probabilities = torch.sigmoid(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, teacher_y).item()
    prediction = probabilities >= args.threshold

    print(f"held-out BCE: {bce:.6f}")
    for index, label in enumerate(("Z correction", "X correction", "residual X", "residual Z")):
        precision, recall, accuracy = binary_metrics(prediction[:, index], teacher_y[:, index])
        print(f"{label}: precision={precision:.4f} recall={recall:.4f} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
