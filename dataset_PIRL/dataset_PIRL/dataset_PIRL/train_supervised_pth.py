#!/usr/bin/env python3
"""
train_supervised_pth_ver2.py
Train RLlib-style ShallowUQNet wrapper as plain nn.Module and save states.
"""
import os, time, importlib.util, numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# USER EDITS
DATA_PATH = "adj_torch_tr100_te10/train_adj_torch_N256_20260202_014420.pt"
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 60
MODEL_BASE_PTH = "shallowuqnet_base_state_0.1k.pth"
MODEL_WRAPPER_PTH = "shallowuqnet_wrapper_state_0.1k.pth"
MODEL_PY = os.path.join(os.path.dirname(__file__), "model.py")

# device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    try:
        is_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    except Exception:
        is_mps = getattr(torch, "has_mps", False) and getattr(torch, "has_mps", False)
    DEVICE = "mps" if is_mps else "cpu"
print("Device:", DEVICE)

# load dataset
data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
if isinstance(data, dict):
    if "dataset" in data and isinstance(data["dataset"], TensorDataset):
        X, Y = data["dataset"].tensors
    elif "X" in data and "Y" in data:
        X, Y = data["X"], data["Y"]
    else:
        raise RuntimeError("Unrecognized .pt format")
else:
    if isinstance(data, TensorDataset):
        X, Y = data.tensors
    else:
        raise RuntimeError("Unrecognized .pt content")

X = X.float(); Y = Y.float()
if X.ndim == 4 and X.shape[2] == 1:
    X = X.squeeze(2)
if X.ndim == 2:
    X = X[:, None, :]

print("Loaded X.shape", X.shape, "Y.shape", Y.shape)
ncells = X.shape[-1]
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# import model
spec = importlib.util.spec_from_file_location("user_model", MODEL_PY)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for cname in ("ShallowUQNet", "FCNQNet", "FCNQNet_heavy"):
    if hasattr(mod, cname):
        ModelClass = getattr(mod, cname); break
else:
    raise RuntimeError("Could not find suitable model class in model.py")

# wrapper
class WrapperModel(nn.Module):
    def __init__(self, base_cls, ncells):
        super().__init__()
        try:
            import gym
            from gym.spaces import Box, Discrete
            obs_space = Box(low=-1.0, high=1.0, shape=(1, ncells), dtype=np.float32)
            action_space = Discrete(2)
        except Exception:
            class DummyBox: 
                def __init__(self, *a, **k): pass
            class DummyDisc:
                def __init__(self, *a, **k): pass
            obs_space = DummyBox(); action_space = DummyDisc()
        num_outputs = ncells; model_config = {}; name = "tmp_model"
        self.base = base_cls(obs_space, action_space, num_outputs, model_config, name)
    def forward(self, x):
        input_dict = {"obs": x}
        out = self.base(input_dict, [], None)
        if isinstance(out, (tuple, list)): out = out[0]
        return out

model = WrapperModel(ModelClass, ncells).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

start = time.time()
for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0; n=0
    for xb, yb in loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        running += float(loss.item()) * xb.shape[0]; n += xb.shape[0]
    epoch_loss = running / max(1,n)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS}  train_loss={epoch_loss:.6e}")
print("Training done in", time.time()-start, "s")

# save
torch.save(model.base.state_dict(), MODEL_BASE_PTH)
print("Saved base state_dict to", MODEL_BASE_PTH)
torch.save(model.state_dict(), MODEL_WRAPPER_PTH)
print("Saved wrapper state_dict to", MODEL_WRAPPER_PTH)
