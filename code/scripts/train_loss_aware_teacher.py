#!/usr/bin/env python3
"""Minimal supervised trainer for a fixed v2 local-teacher shard.

This is a CPU-friendly smoke trainer.  For a real run, generate larger d=9
shards and invoke this script with ``--device cuda`` on Brev.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.predecoder import PreDecoderModelMemory_v2


def make_config(distance: int, rounds: int) -> SimpleNamespace:
    return SimpleNamespace(
        distance=distance,
        n_rounds=rounds,
        model=SimpleNamespace(
            input_channels=5,
            out_channels=4,
            dropout_p=0.0,
            activation="gelu",
            num_filters=[128, 128, 128, 4],
            kernel_size=[3, 3, 3, 3],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    with np.load(args.dataset) as shard:
        train_x = torch.from_numpy(shard["train_x"]).float()
        teacher_y = torch.from_numpy(shard["local_teacher_y"]).float()
        distance = int(shard["distance"])
        rounds = int(shard["n_rounds"])
    if train_x.shape[1] != 5 or teacher_y.shape[1] != 4:
        raise ValueError("dataset must contain five-channel train_x and four-channel local_teacher_y.")

    model = PreDecoderModelMemory_v2(make_config(distance, rounds)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(train_x, teacher_y), batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.shape[0]
        print(f"epoch {epoch + 1}/{args.epochs}: BCE={running_loss / len(train_x):.6f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
