#!/usr/bin/env python3
"""
generate_adj_dataset_torch.py
Generate adjoint dataset using MeeTorch (Torch backend).
Saves {'dataset','X','Y','raw_delta','meta'} in a .pt file.
"""
import os, time, argparse, math
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset
import meent

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=64)
parser.add_argument('--train_size', type=int, default=50)
parser.add_argument('--test_size', type=int, default=10)
parser.add_argument('--wavelength', type=float, default=1100.0)
parser.add_argument('--angle', type=float, default=60.0)
parser.add_argument('--thickness_layers', type=int, default=8)
parser.add_argument('--variable_layer', type=int, default=2)
parser.add_argument('--n_ridge', type=float, default=None)
parser.add_argument('--n_groove', type=float, default=1.0)
parser.add_argument('--out_dir', type=str, default='./adj_torch')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--order_idx', type=int, default=None)
parser.add_argument('--period', type=float, default=None)
parser.add_argument('--fto', type=int, default=40)
parser.add_argument('--dump_config', action='store_true')
args = parser.parse_args()

N = args.N
TRAIN_SIZE = args.train_size
TEST_SIZE = args.test_size
WAVELENGTH = args.wavelength
ANGLE = args.angle
THK = [325] * args.thickness_layers
VAR_LAYER = args.variable_layer
OUT_DIR = args.out_dir
ORDER_IDX = args.order_idx
PERIOD = args.period
FTO = args.fto

os.makedirs(OUT_DIR, exist_ok=True)

# try material table
if args.n_ridge is None:
    try:
        from meent.on_numpy.modeler.modeling import read_material_table, find_nk_index
        mat_table = read_material_table()
        n_ridge = find_nk_index('p_si__real', mat_table, WAVELENGTH)
        print("Found n_ridge from material table:", n_ridge)
    except Exception as e:
        print("Material table not found or error:", e)
        n_ridge = 3.5
else:
    n_ridge = float(args.n_ridge)
n_groove = float(args.n_groove)
DELTA_N_MAG = float(n_ridge - n_groove)
print(f"Using Δn magnitude (n_ridge - n_groove) = {DELTA_N_MAG:.6f}")

if PERIOD is not None:
    period = [float(PERIOD)]
else:
    period = [abs(WAVELENGTH / math.sin(math.radians(ANGLE)))]

if args.dump_config:
    print("CONFIG SUMMARY:")
    print(f"  wavelength = {WAVELENGTH} nm")
    print(f"  period     = {period[0]} nm")
    print(f"  fto        = {FTO}")
    print(f"  ORDER_IDX  = {ORDER_IDX}")

def build_ucell_from_pattern_torch(pattern):
    u = (pattern + 1) / 2.0
    ucell = np.ones((len(THK), 1, N), dtype=np.float32) * n_groove
    ucell[0:2, 0, :] = 1.45
    ucell[VAR_LAYER, 0, :] = u * (n_ridge - n_groove) + n_groove
    return torch.tensor(ucell, dtype=torch.float32, requires_grad=True)

def process_one(pattern_np):
    ucell_t = build_ucell_from_pattern_torch(pattern_np)
    mee = meent.call_mee(backend=2, wavelength=WAVELENGTH, period=period,
                         n_top=1.45, n_bot=1.0, theta=0, phi=0, psi=0,
                         fto=FTO, pol=1, thickness=THK, ucell=ucell_t)
    res = mee.conv_solve()
    assert hasattr(res, 'de_ti'), "res.de_ti missing"
    de = res.de_ti.squeeze()
    # pick order
    if ORDER_IDX is not None:
        order_idx = int(ORDER_IDX)
        try:
            if isinstance(de, torch.Tensor):
                eta = de[order_idx]
            else:
                de_np = np.array(de).squeeze()
                eta = float(de_np[order_idx]); eta = torch.tensor(eta, dtype=torch.float32)
        except Exception:
            # fallback auto
            ORDER = None
    if isinstance(de, torch.Tensor):
        order_idx = int(torch.argmax(de).item())
        eta = de[order_idx]
    else:
        de_np = np.array(de).squeeze()
        order_idx = int(np.argmax(de_np))
        eta = float(de_np[order_idx]); eta = torch.tensor(eta, dtype=torch.float32)
    if ucell_t.grad is not None:
        ucell_t.grad.zero_()
    if isinstance(eta, torch.Tensor) and eta.requires_grad:
        eta.backward()
    else:
        try:
            torch.tensor(float(eta), requires_grad=True).backward()
        except Exception:
            pass
    if ucell_t.grad is None:
        raise RuntimeError("ucell.grad is None — autograd didn't propagate.")
    grad_layer = ucell_t.grad[VAR_LAYER, 0, :].detach().cpu().numpy()
    delta_n_per_cell = -pattern_np * DELTA_N_MAG
    delta_eta = grad_layer * delta_n_per_cell
    norm = np.linalg.norm(delta_eta)
    delta_eta_norm = delta_eta / norm if norm > 0 else delta_eta.copy()
    X = pattern_np.astype(np.float32).reshape(1,1,-1)
    Y_norm = delta_eta_norm.astype(np.float32)
    return X, Y_norm, delta_eta

def generate_split(num, prefix='train', seed0=None):
    rng = np.random.RandomState(seed0 or int(time.time()))
    Xs, Ys, raw_list = [], [], []
    for i in range(num):
        p = rng.choice([1, -1], size=(N,))
        X, Ynorm, raw = process_one(p)
        Xs.append(X); Ys.append(Ynorm); raw_list.append(raw)
        if (i+1) % 10 == 0 or (i+1)==num:
            print(f"{prefix}: generated {i+1}/{num}")
    X_arr = np.vstack(Xs)
    Y_arr = np.vstack(Ys)
    raw_arr = np.vstack(raw_list)
    # ensure shapes (num,1,N)
    if X_arr.ndim == 4 and X_arr.shape[2] == 1:
        X_arr = X_arr.squeeze(2)
    X_t = torch.from_numpy(X_arr).float()
    Y_t = torch.from_numpy(Y_arr).float()
    raw_t = torch.from_numpy(raw_arr).float()
    ds = TensorDataset(X_t, Y_t)
    tnow = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"{prefix}_adj_torch_N{N}_{tnow}.pt")
    metadata = {
        'wavelength': WAVELENGTH,
        'angle': ANGLE,
        'N': N,
        'period': period[0],
        'fto': FTO,
        'thickness_layers': len(THK),
        'variable_layer': VAR_LAYER,
        'n_ridge': n_ridge,
        'n_groove': n_groove,
        'delta_n_magnitude': DELTA_N_MAG,
        'created': tnow,
        'script': __file__
    }
    torch.save({'dataset': ds, 'X': X_t, 'Y': Y_t, 'raw_delta': raw_t, 'meta': metadata}, out_path)
    print("Saved", out_path)
    return out_path

if __name__ == "__main__":
    print("Generating train and test splits ...")
    train_path = generate_split(TRAIN_SIZE, 'train', seed0=args.seed)
    test_path = generate_split(TEST_SIZE, 'test', seed0=(args.seed or int(time.time()))+999)
    print("Done.")
