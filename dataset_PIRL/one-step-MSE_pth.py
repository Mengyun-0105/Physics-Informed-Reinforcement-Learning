#!/usr/bin/env python3
"""
one-step-MSE_pth_ver2.py - quick single-batch MSE check using saved .pth
"""
import os, importlib.util
import torch, numpy as np

DATA_PATH = "adj_torch_tr100_te10/train_adj_torch_N256_20260202_014420.pt"
MODEL_PTH = "shallowuqnet_base_state_0.1k.pth"  # or wrapper pth
MODEL_PY = "model.py"
BATCH_SIZE = 8

data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
if isinstance(data, dict):
    if "X" in data and "Y" in data:
        X, Y = data["X"], data["Y"]
    elif "dataset" in data and isinstance(data["dataset"], torch.utils.data.TensorDataset):
        X, Y = data["dataset"].tensors
    else:
        raise RuntimeError("Unrecognized .pt content")
else:
    X, Y = data.tensors

if X.ndim == 4 and X.shape[2] == 1:
    X = X.squeeze(2)
if X.ndim == 2:
    X = X[:, None, :]

X = X.float(); Y = Y.float()

spec = importlib.util.spec_from_file_location("user_model", os.path.abspath(MODEL_PY))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ModelClass = getattr(mod, "ShallowUQNet")

ncells = X.shape[-1]
try:
    import gym
    from gym.spaces import Box, Discrete
    obs_space = Box(low=-1.0, high=1.0, shape=(1, ncells), dtype=np.float32)
    action_space = Discrete(2)
except Exception:
    class Box: 
        def __init__(self,*a,**k): pass
    class Discrete:
        def __init__(self,*a,**k): pass
    obs_space = Box(); action_space = Discrete(2)

model = ModelClass(obs_space, action_space, ncells, {}, "temp_model")
model.eval()

st = torch.load(MODEL_PTH, map_location='cpu')
loaded = False
try:
    if hasattr(model, "base"):
        model.base.load_state_dict(st); wrapped = model.base; loaded = True
except Exception:
    pass
if not loaded:
    try:
        model.load_state_dict(st); wrapped = model; loaded = True
    except Exception as e:
        raise RuntimeError("Failed to load state into model/base: " + str(e))
wrapped.eval()

with torch.no_grad():
    xb = X[:BATCH_SIZE]
    input_dict = {"obs": xb}
    out = wrapped(input_dict, [], None)
    if isinstance(out, (tuple, list)): out = out[0]
    out = out.reshape(out.shape[0], -1)
    y_true = Y[:out.shape[0]]
    mse = torch.nn.functional.mse_loss(out, y_true)
    print("one-step MSE:", float(mse))
