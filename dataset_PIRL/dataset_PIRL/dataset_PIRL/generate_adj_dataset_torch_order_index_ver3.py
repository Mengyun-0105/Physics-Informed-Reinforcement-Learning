#!/usr/bin/env python3

### Quick change to save an RLlib-compatible file
"""
generate_adj_dataset_torch.py

Generate a dataset of (structure -> normalized adjoint flip-approx Δη) using MeeTorch autograd.
Saves torch.TensorDataset and metadata dict into a .pt file.

Usage (example):
  python generate_adj_dataset_torch.py --N 64 --train_size 100 --out_dir ./adj_torch --wavelength 1100 --angle 60

Notes:
- This uses MeeTorch autograd; it creates one mee per sample and calls backward() once per sample.
- By default it uses the variable layer index 2 (matching your earlier ucell layout).
"""

import os, time, argparse, math
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset
import meent

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=64)
parser.add_argument('--train_size', type=int, default=4)
parser.add_argument('--test_size', type=int, default=2)
parser.add_argument('--wavelength', type=float, default=1100.0)
parser.add_argument('--angle', type=float, default=60.0)
parser.add_argument('--thickness_layers', type=int, default=8)
parser.add_argument('--variable_layer', type=int, default=2,
                    help='index of layer that encodes the pattern (0-based)')
parser.add_argument('--n_ridge', type=float, default=None,
                    help='optional: override refractive index for ridge (Si). If None, will try material table)')
parser.add_argument('--n_groove', type=float, default=1.0)
parser.add_argument('--out_dir', type=str, default='./adj_torch')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--order_idx', type=int, default=None,
                    help='(optional) internal meent order index to use for eta/grad extraction; overrides auto-selection')

parser.add_argument('--period', type=float, default=None,
                    help='optional: grating period Λ in nm. If provided, overrides period computed from --angle')
parser.add_argument('--fto', type=int, default=40,
                    help='fourier truncation / number of harmonics passed to meent (increase for high-order diffraction)')

parser.add_argument('--dump_config', action='store_true',
                    help='print resolved wavelength, period, fto, and order index before running')



args = parser.parse_args()

# params
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

# Attempt to read n_ridge from meent material table if not provided
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

#period = [abs(WAVELENGTH / math.sin(math.radians(ANGLE)))]
# decide period (list form expected by meent.call_mee)
if PERIOD is not None:
    period = [float(PERIOD)]
else:
    # compute period from angle as before (guard against invalid angle)
    period = [abs(WAVELENGTH / math.sin(math.radians(ANGLE)))]

# optional: print configuration summary
if args.dump_config:
    print("CONFIG SUMMARY:")
    print(f"  wavelength = {WAVELENGTH} nm")
    print(f"  period     = {period[0]} nm")
    print(f"  fto        = {FTO}")
    print(f"  ORDER_IDX  = {ORDER_IDX}")


def build_ucell_from_pattern_torch(pattern):
    """pattern: numpy array of shape (N,) with values +1/-1"""
    # u = (p + 1)/2
    u = (pattern + 1) / 2.0  # 0..1
    # build numpy ucell then convert to torch to pass into meent (MeeTorch accepts torch tensor)
    ucell = np.ones((len(THK), 1, N), dtype=np.float32) * n_groove
    # fill top layers with top index
    ucell[0:2, 0, :] = 1.45
    # variable layer
    ucell[VAR_LAYER, 0, :] = u * (n_ridge - n_groove) + n_groove
    return torch.tensor(ucell, dtype=torch.float32, requires_grad=True)

def process_one(pattern_np):
    """
    pattern_np: numpy array shape (N,) values +1/-1
    returns X (1,1,N) float32, Y_norm (N,) float32, raw_delta_eta (N,)
    """
    # create torch ucell (requires_grad True)
    ucell_t = build_ucell_from_pattern_torch(pattern_np)

    # create mee with torch backend (it accepted torch ucell in your test)
    mee = meent.call_mee(backend=2, wavelength=WAVELENGTH, period=period,
                         n_top=1.45, n_bot=1.0, theta=0, phi=0, psi=0,
                         fto=FTO, pol=1, thickness=THK, ucell=ucell_t)

    # forward solve
    res = mee.conv_solve()
    assert hasattr(res, 'de_ti'), "res.de_ti missing"

    # choose order (dominant transmitted order) or override with CLI
    de = res.de_ti.squeeze()
    # If user provided ORDER_IDX, use that (validate bounds)
    if ORDER_IDX is not None:
        try:
            order_idx = int(ORDER_IDX)
            # basic bounds check using de length (works for torch tensor or numpy)
            de_len = de.shape[0] if hasattr(de, 'shape') else len(np.array(de).squeeze())
            if not (0 <= order_idx < int(de_len)):
                print(f"[warning] ORDER_IDX {order_idx} out of bounds (0..{int(de_len)-1}), falling back to auto-selection")
                order_idx = None
        except Exception:
            print("[warning] invalid ORDER_IDX provided; falling back to auto-selection")
            order_idx = None
    else:
        order_idx = None

    # If ORDER_IDX not used/valid, auto-select dominant transmitted order (existing behavior)
    if order_idx is None:
        if isinstance(de, torch.Tensor):
            order_idx = int(torch.argmax(de).item())
            eta = de[order_idx]
        else:
            de_np = np.array(de).squeeze()
            order_idx = int(np.argmax(de_np))
            eta = float(de_np[order_idx])
            eta = torch.tensor(eta, dtype=torch.float32)
    else:
        # ORDER_IDX is valid and set: pick eta from de accordingly
        if isinstance(de, torch.Tensor):
            eta = de[order_idx]
        else:
            de_np = np.array(de).squeeze()
            eta = float(de_np[order_idx])
            eta = torch.tensor(eta, dtype=torch.float32)

    # backward to get gradient wrt ucell
    # Zero grads not necessary because using a fresh ucell, but be safe:
    if ucell_t.grad is not None:
        ucell_t.grad.zero_()
    # If eta is torch scalar with grad history:
    if isinstance(eta, torch.Tensor) and eta.requires_grad:
        eta.backward()
    else:
        # If not, convert to tensor and call backward (this may not attach to ucell)
        try:
            torch.tensor(float(eta), requires_grad=True).backward()
        except Exception:
            pass

    # get gradient tensor for the variable layer
    if ucell_t.grad is None:
        raise RuntimeError("ucell.grad is None — autograd didn't propagate. Check MeeTorch API.")
    grad_layer = ucell_t.grad[VAR_LAYER, 0, :].detach().cpu().numpy()  # shape (N,)

    # compute per-cell Δη_approx = grad * Δn_cell
    # Δn_cell = -p * (n_ridge - n_groove)
    delta_n_per_cell = -pattern_np * DELTA_N_MAG
    delta_eta = grad_layer * delta_n_per_cell  # numpy array

    # normalize L2
    norm = np.linalg.norm(delta_eta)
    if norm > 0:
        delta_eta_norm = delta_eta / norm
    else:
        delta_eta_norm = delta_eta.copy()

    # X format (1,1,N)
    X = ((pattern_np.astype(np.float32)).reshape(1,1,-1))
    Y_norm = delta_eta_norm.astype(np.float32)

    return X, Y_norm, delta_eta

def generate_split(num, prefix='train', seed0=None):
    rng = np.random.RandomState(seed0 or int(time.time()))
    Xs, Ys, raw_list = [], [], []
    for i in range(num):
        p = rng.choice([1, -1], size=(N,))
        X, Ynorm, raw = process_one(p)
        Xs.append(X)
        Ys.append(Ynorm)
        raw_list.append(raw)
        if (i+1) % 10 == 0 or (i+1)==num:
            print(f"{prefix}: generated {i+1}/{num}")
    X_arr = np.vstack(Xs)   # (num,1,1,N)
    Y_arr = np.vstack(Ys)   # (num,N)
    raw_arr = np.vstack(raw_list)  # (num,N)
    ds = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(Y_arr))
    tnow = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"{prefix}_adj_torch_N{N}_{tnow}.pt")
    # Save both dataset and metadata (and raw deltas)
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
    #torch.save({'dataset': ds, 'raw_delta': torch.from_numpy(raw_arr), 'meta': metadata}, out_path)


    # Ensure shapes: X_arr came out (num,1,1,N) in some variants; we want (num,1,N)
    if X_arr.ndim == 4 and X_arr.shape[2] == 1:
        X_arr = X_arr.squeeze(2)   # -> (num,1,N)

    # ensure types
    X_t = torch.from_numpy(X_arr).float()
    Y_t = torch.from_numpy(Y_arr).float()
    raw_t = torch.from_numpy(raw_arr).float()

    # Create a TensorDataset that many tools expect
    ds = TensorDataset(X_t, Y_t)

    # Save: include both 'dataset' (TensorDataset) and raw tensors for inspection
    out_path = os.path.join(OUT_DIR, f"{prefix}_adj_torch_N{N}_{tnow}.pt")
    torch.save({
        'dataset': ds,
        'X': X_t,
        'Y': Y_t,
        'raw_delta': raw_t,
        'meta': metadata
    }, out_path)
    print("Saved", out_path)

if __name__ == "__main__":
    print("Generating train and test splits ...")
    train_path = generate_split(TRAIN_SIZE, 'train', seed0=args.seed)
    test_path = generate_split(TEST_SIZE, 'test', seed0=(args.seed or int(time.time()))+999)
    print("Done.")
