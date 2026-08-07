#!/usr/bin/env python3
"""Evaluate final logical parity from a residual global MLP."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from scripts.train_bluvstein_global_decoder import BluvsteinResidualMLP, features

def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset",type=Path); p.add_argument("checkpoint",type=Path); p.add_argument("--device",default="cpu"); args=p.parse_args()
    with np.load(args.dataset) as d:
        x=torch.from_numpy(features(d)).float(); target=torch.from_numpy(d["target"]).float(); obs=d["observables"].astype(np.uint8); frame=d["local_frame"].astype(np.uint8)
    device=torch.device(args.device); ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=BluvsteinResidualMLP(int(ckpt["input_dim"]),int(ckpt["output_dim"])).to(device); model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad(): logits=model(x.to(device)).cpu(); bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,target).item(); residual=(torch.sigmoid(logits)>=.5).numpy().astype(np.uint8)
    final=residual ^ frame[:,None]; print(f"BCE: {bce:.6f}"); print(f"global residual accuracy: {(residual == target.numpy()).mean():.6f}"); print(f"final logical LER: {np.any(final != obs,axis=1).mean():.6g}")
if __name__ == "__main__": main()
