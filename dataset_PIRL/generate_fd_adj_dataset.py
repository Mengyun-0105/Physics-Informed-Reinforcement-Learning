#!/usr/bin/env python3
"""
generate_fd_adj_dataset.py
Finite-difference dataset generator (flip each cell) -> normalized delta-eta per cell.
Run: python generate_fd_adj_dataset.py --N 64 --train_size 4 --test_size 2 ...
"""
import os, time, math, argparse
from datetime import datetime
from multiprocessing import Pool
from functools import partial

import numpy as np
import torch
from torch.utils.data import TensorDataset
import meent

parser = argparse.ArgumentParser()
parser.add_argument('--wavelength', type=float, default=1100.0)
parser.add_argument('--angle', type=float, default=60.0)
parser.add_argument('--N', type=int, default=64)
parser.add_argument('--train_size', type=int, default=4)
parser.add_argument('--test_size', type=int, default=2)
parser.add_argument('--coarsen', type=int, default=1)
parser.add_argument('--out_dir', type=str, default='./adj_fd_data')
parser.add_argument('--checkpoint_interval', type=int, default=2)
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--thickness_layers', type=int, default=8)
parser.add_argument('--mfs_width', type=int, default=None)
args = parser.parse_args()

WAVELENGTH = args.wavelength
ANGLE = args.angle
N_CELLS = args.N
TRAIN_SIZE = args.train_size
TEST_SIZE = args.test_size
COARSEN = args.coarsen
OUT_DIR = args.out_dir
WORKERS = max(1, args.workers)
THICKNESS_LIST = [325] * args.thickness_layers
n_top = 1.45
n_bot = 1.0
DELTA_N = None

os.makedirs(OUT_DIR, exist_ok=True)

def build_ucell_from_binary(struct_bin, wavelength=WAVELENGTH):
    try:
        from meent.on_numpy.modeler.modeling import read_material_table, find_nk_index
        mat_table = read_material_table()
        n_ridge = find_nk_index('p_si__real', mat_table, wavelength)
        n_groove = 1.0
    except Exception:
        n_ridge = 3.5
        n_groove = 1.0

    global DELTA_N
    if DELTA_N is None:
        DELTA_N = float(n_ridge - n_groove)

    u = (struct_bin + 1) / 2.0
    ucell = np.ones((len(THICKNESS_LIST), 1, struct_bin.size), dtype=float) * n_groove
    ucell[0:2, 0, :] = n_top
    ucell[2, 0, :] = u * (n_ridge - n_groove) + n_groove
    return ucell

def make_mee(ucell, wavelength=WAVELENGTH):
    period = [abs(wavelength / math.sin(math.radians(ANGLE)))]
    mee = meent.call_mee(
        backend=0, wavelength=wavelength, period=period,
        n_top=n_top, n_bot=n_bot, theta=0, phi=0, psi=0,
        fto=40, pol=1, thickness=THICKNESS_LIST, ucell=ucell
    )
    return mee

def extract_eta_from_result(mee_obj, res_obj, order_idx, verbose=False):
    import numpy as np
    try:
        if hasattr(res_obj, 'de_ti'):
            arr = np.array(res_obj.de_ti)
            arr1 = arr.squeeze()
            if arr1.ndim == 1 and order_idx < arr1.shape[0]:
                if verbose:
                    print("extract_eta_from_result: using res.de_ti (squeezed) - shape", arr.shape)
                return float(arr1[order_idx])
    except Exception as e:
        if verbose:
            print("extract_eta_from_result: de_ti extraction failed:", e)

    try:
        if hasattr(res_obj, 'res_te_inc'):
            amps = np.array(res_obj.res_te_inc)
            try:
                amps_sel = amps[..., order_idx]
                eta = float(np.sum(np.abs(amps_sel)**2))
                if verbose:
                    print("extract_eta_from_result: using res.res_te_inc (last-axis). amps.shape:", amps.shape)
                return eta
            except Exception:
                pass
            try:
                amps_sel = amps[order_idx]
                eta = float(np.sum(np.abs(amps_sel)**2))
                if verbose:
                    print("extract_eta_from_result: using res.res_te_inc (first-axis). amps.shape:", amps.shape)
                return eta
            except Exception:
                pass
    except Exception as e:
        if verbose:
            print("extract_eta_from_result: res_te_inc fallback failed:", e)

    if verbose:
        print("extract_eta_from_result: WARNING - could not extract eta; returning 0.0")
    return 0.0

def process_one_structure(idx_seed_tuple, N=N_CELLS, coarsen=COARSEN, wavelength=WAVELENGTH, angle_deg=ANGLE):
    idx, seed = idx_seed_tuple
    rng = np.random.RandomState(seed)
    struct = rng.choice([1, -1], size=(N,))
    if coarsen > 1:
        newN = N // coarsen
        struct_coarse = np.zeros(newN, dtype=int)
        for i in range(newN):
            block = struct[i*coarsen:(i+1)*coarsen]
            struct_coarse[i] = 1 if np.sum(block) > 0 else -1
        struct = struct_coarse

    ucell = build_ucell_from_binary(struct, wavelength=wavelength)
    mee = make_mee(ucell, wavelength=wavelength)
    res = mee.conv_solve()

    order_idx = 1
    try:
        if hasattr(res, 'de_ti'):
            de = np.array(res.de_ti).squeeze()
            if de.ndim == 1 and de.size > 0:
                order_idx = int(np.argmax(de))
        else:
            if hasattr(mee, 'get_kx_ky_vector'):
                try:
                    kx_vec, ky_vec = mee.get_kx_ky_vector(wavelength)
                    kx = np.array(kx_vec).reshape(-1)
                    k0 = 2 * math.pi / wavelength
                    angs = []
                    for k in kx:
                        try:
                            angs.append(math.degrees(math.asin(float(k) / k0)))
                        except Exception:
                            angs.append(None)
                    diffs = [abs(a - angle_deg) if a is not None else 1e9 for a in angs]
                    order_idx = int(np.argmin(diffs))
                except Exception:
                    order_idx = 1
    except Exception:
        order_idx = 1

    if idx < 2:
        print(f"[debug] idx={idx} chosen order_idx={order_idx}")
        if hasattr(res, 'de_ti'):
            import numpy as _np
            print(f"[debug] de_ti (squeezed) sample: {_np.array(res.de_ti).squeeze()[:10]} ...")

    eta0 = extract_eta_from_result(mee, res, order_idx, verbose=(idx < 2))

    M = struct.shape[0]
    delta_eta = np.zeros(M, dtype=float)

    for i in range(M):
        struct2 = struct.copy()
        struct2[i] = -struct2[i]
        ucell2 = build_ucell_from_binary(struct2, wavelength=wavelength)
        mee2 = make_mee(ucell2, wavelength=wavelength)
        res2 = mee2.conv_solve()
        eta_i = extract_eta_from_result(mee2, res2, order_idx, verbose=(idx < 1 and i < 2))
        delta_eta[i] = eta_i - eta0

    norm = np.linalg.norm(delta_eta)
    if norm > 0:
        delta_eta_norm = delta_eta / norm
    else:
        delta_eta_norm = delta_eta

    X = struct.reshape(1,1,-1).astype(np.float32)
    Y = delta_eta_norm.astype(np.float32)

    return X, Y

def build_dataset(num_samples, prefix='train'):
    seeds = [int(time.time()) + i*37 for i in range(num_samples)]
    out_X = []
    out_Y = []
    if WORKERS > 1:
        with Pool(processes=WORKERS) as pool:
            for i, (X,Y) in enumerate(pool.imap_unordered(partial(process_one_structure, N=N_CELLS, coarsen=COARSEN), enumerate(seeds))):
                out_X.append(X); out_Y.append(Y)
                if (i+1) % args.checkpoint_interval == 0 or (i+1)==num_samples:
                    save_checkpoint(out_X, out_Y, prefix)
    else:
        for i, seed in enumerate(seeds):
            X,Y = process_one_structure((i,seed), N=N_CELLS, coarsen=COARSEN)
            out_X.append(X); out_Y.append(Y)
            if (i+1) % args.checkpoint_interval == 0 or (i+1)==num_samples:
                save_checkpoint(out_X, out_Y, prefix)

    X_arr = np.vstack(out_X)
    Y_arr = np.vstack(out_Y)
    ds = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(Y_arr))
    tnow = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"{prefix}_adj_fd_N{N_CELLS}_coarsen{COARSEN}_{tnow}.pt")
    torch.save(ds, out_path)
    print("Saved dataset to", out_path)
    return out_path

def save_checkpoint(Xlist, Ylist, prefix):
    if len(Xlist) == 0:
        return
    X_arr = np.vstack(Xlist)
    Y_arr = np.vstack(Ylist)
    ds = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(Y_arr))
    tnow = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"{prefix}_partial_N{N_CELLS}_coarsen{COARSEN}_{len(Xlist)}_{tnow}.pt")
    torch.save(ds, out_path)
    print("Checkpoint saved:", out_path)

if __name__ == "__main__":
    print("Running finite-difference dataset generator.")
    print(f"N={N_CELLS}, train_size={TRAIN_SIZE}, coarsen={COARSEN}, workers={WORKERS}")
    t0 = time.time()
    train_path = build_dataset(TRAIN_SIZE, prefix='train')
    test_path = build_dataset(TEST_SIZE, prefix='test')
    print("All done. Time:", time.time()-t0)
