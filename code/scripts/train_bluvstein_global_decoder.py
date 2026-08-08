#!/usr/bin/env python3
"""Train a global loss-aware MLP on frozen-predecoder residuals."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BluvsteinResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(),
                                 nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(),
                                 nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Linear(256, output_dim))
    def forward(self, x): return self.net(x)


def features(shard):
    core = np.concatenate([shard["residual"], shard["heralded_erasures"][:, None]], axis=1).reshape(len(shard["target"]), -1)
    readout = shard["logical_readout"].reshape(-1, 1) if "logical_readout" in shard else np.zeros((len(core), 1), dtype=np.uint8)
    raw = shard["raw_measurements"] if "raw_measurements" in shard else np.empty((len(core), 0), dtype=np.uint8)
    return np.concatenate([core, raw, readout], axis=1)


def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset", type=Path); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu"); p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch-size", type=int, default=256); p.add_argument("--lr", type=float, default=3e-4); args=p.parse_args()
    with np.load(args.dataset) as d: x=torch.from_numpy(features(d)).float(); y=torch.from_numpy(d["target"]).float()
    device=torch.device(args.device); model=BluvsteinResidualMLP(x.shape[1], y.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr); loss_fn=nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        total=0.0
        for xb,yb in DataLoader(TensorDataset(x,y),batch_size=args.batch_size,shuffle=True):
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb.to(device)),yb.to(device)); loss.backward(); opt.step(); total+=loss.item()*len(xb)
        print(f"epoch {epoch+1}/{args.epochs}: BCE={total/len(x):.6f}")
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"input_dim":x.shape[1],"output_dim":y.shape[1]},args.output); print(f"wrote {args.output}")

if __name__ == "__main__": main()
