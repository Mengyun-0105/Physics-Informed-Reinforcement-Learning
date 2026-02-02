#!/usr/bin/env python3
"""
plot_learning_curve.py
Runs repeats across sizes and records normalized RMSE:
Normalized RMSE = sqrt(MSE_on_test) / std_train_adjoints
std_train_adjoints is computed on the training subset 'raw' adjoint gradients if present,
otherwise on the training Y values (fallback).
"""
import os, argparse, time, random, csv, importlib.util
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--train_pt', required=True)
parser.add_argument('--test_pt', required=True)
parser.add_argument('--model_py', default='model.py')
parser.add_argument('--out_dir', default='learning_curve_results')
parser.add_argument('--sizes', type=int, nargs='+', default=[10,25,50,100,250,500,1000,2500])
parser.add_argument('--repeats', type=int, default=5)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--device', default=None)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
if args.device:
    DEVICE = args.device
else:
    if torch.cuda.is_available(): DEVICE = "cuda"
    else:
        try:
            is_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
        except Exception:
            is_mps = getattr(torch, "has_mps", False)
        DEVICE = "mps" if is_mps else "cpu"
print("Using device:", DEVICE)

def load_XY_and_raw(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(d, dict):
        if "X" in d and "Y" in d:
            X = d["X"]; Y = d["Y"]
        elif "dataset" in d and isinstance(d["dataset"], TensorDataset):
            X, Y = d["dataset"].tensors
        else:
            raise RuntimeError("Unrecognized .pt content")
        raw = d.get("raw_delta", None)
    else:
        if isinstance(d, TensorDataset):
            X, Y = d.tensors; raw = None
        else:
            raise RuntimeError("Unrecognized .pt content")
    if X.ndim == 4 and X.shape[2] == 1: X = X.squeeze(2)
    if X.ndim == 2: X = X[:, None, :]
    return X.float(), Y.float(), (raw.float() if raw is not None else None), d.get('meta', None)

print("Loading training data:", args.train_pt)
X_all, Y_all, raw_all, meta_train = load_XY_and_raw(args.train_pt)
N_all = X_all.shape[0]
print("Train size:", N_all, "shape:", X_all.shape)

print("Loading test data:", args.test_pt)
X_test, Y_test, raw_test, meta_test = load_XY_and_raw(args.test_pt)
print("Test size:", X_test.shape[0], "shape:", X_test.shape)

spec = importlib.util.spec_from_file_location("user_model", os.path.abspath(args.model_py))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ModelClass = getattr(mod, "ShallowUQNet")
print("Found model class:", ModelClass.__name__)

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
                def __init__(self,*a,**k): pass
            class DummyDisc:
                def __init__(self,*a,**k): pass
            obs_space = DummyBox(); action_space = DummyDisc()
        self.base = base_cls(obs_space, action_space, ncells, {}, "tmp")

    def forward(self, x):
        input_dict = {"obs": x}
        out = self.base(input_dict, [], None)
        if isinstance(out, (tuple,list)): out = out[0]
        return out

def train_on_subset(train_idx, X_full, Y_full, ncells, epochs=args.epochs, batch_size=args.batch, lr=args.lr):
    Xs = X_full[train_idx]; Ys = Y_full[train_idx]
    ds = TensorDataset(Xs, Ys)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = WrapperModel(ModelClass, ncells).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward(); opt.step()
    return model, Xs, Ys

def evaluate_model_and_compute_norm_rmse(model, X_test, Y_test, X_train_subset, raw_train_subset=None, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):
            xb = X_test[i:i+batch_size].to(DEVICE)
            out = model(xb)
            if isinstance(out, (tuple,list)): out = out[0]
            preds.append(out.cpu())
    preds = torch.cat(preds, dim=0)
    mse = float(torch.nn.functional.mse_loss(preds, Y_test))
    rmse = np.sqrt(mse)
    # compute std of adjoint grads in training set:
    if raw_train_subset is not None:
        std_train = float(raw_train_subset.reshape(-1).std().item())
    else:
        # fallback: use Y values (may be normalized). This is a fallback if raw_delta is not present.
        std_train = float(X_train_subset_or_y_train.std().item()) if False else float(Y_train_subset.reshape(-1).std().item())
    # Avoid divide by zero:
    if std_train == 0:
        norm_rmse = float('inf')
    else:
        norm_rmse = rmse / std_train
    return mse, rmse, std_train, norm_rmse, preds

# main
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
sizes = sorted([s for s in args.sizes if s <= N_all])
results = []  # list: size, repeat, mse, rmse, std_train, norm_rmse
ncells = X_all.shape[-1]
print("Will run sizes:", sizes, "repeats:", args.repeats)

start = time.time()
for s in sizes:
    for r in range(args.repeats):
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] size={s} repeat={r+1}/{args.repeats}")
        idx = np.random.choice(N_all, size=s, replace=False)
        # prepare raw subset if available
        raw_subset = raw_all[idx] if (raw_all is not None) else None
        # train
        model, Xtrain_sub, Ytrain_sub = train_on_subset(idx, X_all, Y_all, ncells, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
        # compute norm rmse: use raw_subset if available; else use Ytrain_sub
        if raw_subset is not None:
            raw_train_sub = raw_subset
        else:
            raw_train_sub = None
        mse, rmse, std_train, norm_rmse, preds = evaluate_model_and_compute_norm_rmse(
            model, X_test, Y_test, Xtrain_sub, raw_train_subset=raw_train_sub)
        print(f" -> mse={mse:.4e} rmse={rmse:.4e} std_train={std_train:.4e} norm_rmse={norm_rmse:.4e}")
        results.append({'size': s, 'repeat': r, 'mse': mse, 'rmse': rmse, 'std_train': std_train, 'norm_rmse': norm_rmse})
        # write temporary CSV
        tmp_csv = os.path.join(args.out_dir, "learning_curve_raw_tmp.csv")
        with open(tmp_csv, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=['size','repeat','mse','rmse','std_train','norm_rmse'])
            writer.writeheader()
            for rr in results: writer.writerow(rr)

dur = time.time()-start
print("All runs done in", dur, "s")

# aggregate per size on normalized RMSE (and also save MSE)
agg = defaultdict(list)
agg_mse = defaultdict(list)
for r in results:
    agg[r['size']].append(r['norm_rmse'])
    agg_mse[r['size']].append(r['mse'])

sizes_sorted = sorted(agg.keys())
means = [np.mean(agg[s]) for s in sizes_sorted]
stds  = [np.std(agg[s]) for s in sizes_sorted]
means_mse = [np.mean(agg_mse[s]) for s in sizes_sorted]
stds_mse  = [np.std(agg_mse[s]) for s in sizes_sorted]

raw_csv = os.path.join(args.out_dir, "learning_curve_raw.csv")
with open(raw_csv, 'w', newline='') as cf:
    writer = csv.DictWriter(cf, fieldnames=['size','repeat','mse','rmse','std_train','norm_rmse'])
    writer.writeheader()
    for rr in results: writer.writerow(rr)

agg_csv = os.path.join(args.out_dir, "learning_curve_agg.csv")
with open(agg_csv, 'w', newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(['size','mean_norm_rmse','std_norm_rmse','n_repeats','mean_mse','std_mse'])
    for s, m, st, mm, sm in zip(sizes_sorted, means, stds, means_mse, stds_mse):
        writer.writerow([s, m, st, len(agg[s]), mm, sm])

# plot normalized RMSE
plt.figure(figsize=(6,4))
plt.errorbar(sizes_sorted, means, yerr=stds, fmt='-o')
plt.xscale('log')
plt.xlabel('Training set size (log scale)')
plt.ylabel('Normalized RMSE (rmse / std_train)')
plt.title('Learning curve (normalized RMSE mean ± std)')
plt.grid(True, which='both', ls='--', alpha=0.5)
png_path = os.path.join(args.out_dir, "learning_curve_norm_rmse.png")
plt.savefig(png_path, dpi=200, bbox_inches='tight')
plt.close()
print("Saved plot:", png_path)
print("Saved raw CSV:", raw_csv)
print("Saved aggregated CSV:", agg_csv)
