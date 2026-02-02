#!/usr/bin/env python3
"""
visualise_predictions_pth_ver2.py
Saves predicted vs true curves (PNGs) using saved wrapper/base state.
"""
import os, importlib.util
import torch, numpy as np, matplotlib.pyplot as plt

DATA_PATH = "adj_torch_tr100_te10/test_adj_torch_N256_20260202_014423.pt"
MODEL_PTH = "shallowuqnet_base_state_0.1k.pth"   # base
MODEL_PY = "model.py"
OUT_DIR = "viz_preds/viz_pred_100_10_base"
BATCH = 8
N_SHOW = 10

os.makedirs(OUT_DIR, exist_ok=True)

# load data
d = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
if isinstance(d, dict):
    if "X" in d and "Y" in d:
        X, Y = d["X"], d["Y"]
    elif "dataset" in d and isinstance(d["dataset"], torch.utils.data.TensorDataset):
        X, Y = d["dataset"].tensors
    else:
        raise RuntimeError("Unrecognized .pt format")
else:
    X, Y = d.tensors

if X.ndim == 4 and X.shape[2] == 1:
    X = X.squeeze(2)
if X.ndim == 2:
    X = X[:, None, :]
X = X.float(); Y = Y.float()

# import & instantiate model
spec = importlib.util.spec_from_file_location("user_model", MODEL_PY)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ModelClass = getattr(mod, "ShallowUQNet")

# build model instance (RLlib style)
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
model = ModelClass(obs_space, action_space, ncells, {}, "viz_model")

# try to load wrapper/base
loaded = False
try:
    model.base.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
    wrapped = model.base
    loaded = True
except Exception:
    try:
        model.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
        wrapped = model
        loaded = True
    except Exception as e:
        raise RuntimeError("Failed to load model weights: " + str(e))
wrapped.eval()

# forward for all test samples
preds = []
with torch.no_grad():
    for i in range(0, X.shape[0], BATCH):
        xb = X[i:i+BATCH]
        input_dict = {"obs": xb}
        out = wrapped(input_dict, [], None)
        if isinstance(out, (tuple,list)): out = out[0]
        preds.append(out.cpu())
preds = torch.cat(preds, dim=0)

def plot_sample(i, x, y_true, y_pred):
    L = x.size if hasattr(x,'size') else len(x)
    fig, axs = plt.subplots(3,1,figsize=(6,6), constrained_layout=True)
    axs[0].stem(np.arange(L), x.flatten(), markerfmt='C0o', basefmt=" ")
    axs[0].set_ylim(-1.5, 1.5)
    axs[0].set_title(f"Input pattern")
    axs[1].plot(np.arange(L), y_true.flatten(), linewidth=1)
    axs[1].set_title("True Δη")
    axs[2].plot(np.arange(L), y_pred.flatten(), linewidth=1)
    axs[2].set_title("Predicted Δη")
    fname = os.path.join(OUT_DIR, f"sample_{i:04d}.png")
    fig.suptitle(f"Sample {i}")
    fig.savefig(fname, dpi=200)
    plt.close(fig)

count = min(N_SHOW, X.shape[0])
for i in range(count):
    xi = X[i].numpy().reshape(-1)
    yi = Y[i].numpy().reshape(-1)
    pi = preds[i].numpy().reshape(-1)
    plot_sample(i, xi, yi, pi)

print(f"Saved {count} PNG figures to {OUT_DIR}/")
