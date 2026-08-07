#!/usr/bin/env python3
"""Freeze a Chamberland CNN and prepare loss-aware global-NN supervision."""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from data.predecoded_features import chamberland_residual_and_frame
from model.predecoder import PreDecoderModelMemory_v1


def config(distance, rounds):
    return SimpleNamespace(distance=distance, n_rounds=rounds, model=SimpleNamespace(
        input_channels=4, out_channels=4, dropout_p=0.0, activation="gelu",
        num_filters=[128, 128, 128, 4], kernel_size=[3, 3, 3, 3]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path); p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256); args = p.parse_args()
    with np.load(args.dataset) as d:
        x = torch.from_numpy(d["train_x"][:, :4]).float(); flags = d["heralded_erasures"].astype(np.uint8)
        obs = d["observables"].astype(np.uint8); distance, rounds = int(d["distance"]), int(d["n_rounds"])
    device = torch.device(args.device); model = PreDecoderModelMemory_v1(config(distance, rounds)).to(device)
    # NVIDIA's public checkpoint predates PyTorch 2.6's weights_only default.
    # It is a trusted state dictionary shipped with the Ising-Decoding project.
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)); model.eval()
    residuals=[]; frames=[]
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            xb=x[start:start+args.batch_size].to(device); out=(torch.sigmoid(model(xb)) >= .5).to(torch.uint8)
            r,f=chamberland_residual_and_frame(xb, out); residuals.append(r.cpu()); frames.append(f.cpu())
    residual=torch.cat(residuals).numpy().astype(np.uint8); frame=torch.cat(frames).numpy().astype(np.uint8)
    # Global target is the logical parity remaining after the local frame.
    target=obs ^ frame[:, None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, residual=residual, heralded_erasures=flags, local_frame=frame,
                        target=target, observables=obs, distance=np.asarray(distance), n_rounds=np.asarray(rounds))
    print(f"wrote {args.output}")
    print(f"residual shape: {residual.shape}; global target shape: {target.shape}")

if __name__ == "__main__": main()
