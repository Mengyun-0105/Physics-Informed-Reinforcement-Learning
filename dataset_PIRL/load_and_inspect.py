#!/usr/bin/env python3
"""
load_and_inspect.py - quick inspector for saved .pt datasets
"""
import torch, numpy as np, os
from torch.utils.data import TensorDataset

PATH = "adj_torch_tr100_te10/train_adj_torch_N256_20260202_014420.pt"  # edit or pass absolute path

data = torch.load(PATH, map_location="cpu", weights_only=False)
print("Loaded keys:", list(data.keys()) if isinstance(data, dict) else type(data))

if isinstance(data, dict):
    meta = data.get("meta", None)
    raw = data.get("raw_delta", None)
    if "dataset" in data and isinstance(data["dataset"], TensorDataset):
        X, Y = data["dataset"].tensors
    else:
        X = data.get("X", None); Y = data.get("Y", None)
else:
    # direct TensorDataset
    if isinstance(data, TensorDataset):
        X, Y = data.tensors
        meta = None; raw = None
    else:
        raise RuntimeError("Unrecognized data format in file")

print("Meta:", meta)
print("X.shape:", X.shape, "Y.shape:", Y.shape)
if raw is not None:
    print("raw_delta shape:", tuple(raw.shape))
    r = raw.numpy()
    print("raw_delta stats min,mean,max:", r.min(), r.mean(), r.max())
# Inspect first sample
x0 = X[0].numpy().reshape(-1)
y0 = Y[0].numpy().reshape(-1)
print("sample0: X unique:", sorted(set(x0.tolist())))
print("sample0: Y stats min,mean,max,norm:", float(y0.min()), float(y0.mean()), float(y0.max()), float((y0**2).sum()**0.5))
