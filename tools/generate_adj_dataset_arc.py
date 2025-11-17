#!/usr/bin/env python3
"""
generate_adj_dataset_torch.py  (ARC-friendly patched version)

Generates (structure -> normalized Δη) using MeeTorch autograd when available
(or finite-difference flips when requested or when Torch backend unavailable).

Saves chunked files as dict {'X','Y','raw_delta','meta'} plus a TensorDataset
under the 'dataset' key for backwards compatibility.

Usage example (interactive GPU node):
  conda activate /data/.../venvs/meent_test
  python generate_adj_dataset_torch.py --N 64 --train_size 8 --test_size 2 \
    --out_dir /data/.../adj_run --device cuda:0 --chunk-size 4 --seed 20251117

Notes:
 - If you want strict FD (flip) values use --fd.
 - If you use array jobs, launch each array task with a distinct seed offset.
"""

import os, time, argparse, math, signal, sys, logging
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset
import meent

# ---------------- CLI ----------------
p = argparse.ArgumentParser()
p.add_argument('--N', type=int, default=64)
p.add_argument('--train_size', type=int, default=4)
p.add_argument('--test_size', type=int, default=2)
p.add_argument('--wavelength', type=float, default=1100.0)
p.add_argument('--angle', type=float, default=60.0)
p.add_argument('--thickness_layers', type=int, default=8)
p.add_argument('--variable_layer', type=int, default=2)
p.add_argument('--n_ridge', type=float, default=None)
p.add_argument('--n_groove', type=float, default=1.0)
p.add_argument('--out_dir', type=str, default='./adj_torch')
p.add_argument('--seed', type=int, default=12345)
p.add_argument('--order_idx', type=int, default=None)
p.add_argument('--period', type=float, default=None)
p.add_argument('--fto', type=int, default=40)
p.add_argument('--ridge_material', type=str, default='Si')
p.add_argument('--groove_material', type=str, default='Air')
p.add_argument('--dump_config', action='store_true')
# HPC-friendly options:
p.add_argument('--device', type=str, default=None, help='cuda:0 or cpu (auto-detect if omitted)')
p.add_argument('--fd', action='store_true', help='force finite-difference flipping instead of autograd')
p.add_argument('--chunk-size', type=int, default=None, help='checkpoint/save size (defaults to train_size)')
p.add_argument('--checkpoint_interval', type=int, default=50, help='how often (samples) to write checkpoint')
p.add_argument('--out_prefix', type=str, default=None)
p.add_argument('--workers', type=int, default=1, help='threads to use (passed to torch.set_num_threads)')
args = p.parse_args()

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('gen_adj')

# ---------------- params ----------------
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
SEED_BASE = int(args.seed)

os.makedirs(OUT_DIR, exist_ok=True)

# chunk default
if args.chunk_size is None:
    args.chunk_size = TRAIN_SIZE

# threads
torch.set_num_threads(max(1, args.workers))

# ---------------- material helper (same logic you had) ----------------
# uses materials.py mapping if available; fallback numeric table
try:
    from materials import MAPPING, FALLBACK_N
except Exception:
    MAPPING = {}
    FALLBACK_N = {}

def _is_number_like(s):
    try:
        float(s)
        return True
    except Exception:
        return False

def resolve_n_for_material(material_name, wavelength):
    if material_name is None:
        return 1.0
    if isinstance(material_name, (int, float)) or _is_number_like(material_name):
        return float(material_name)

    meent_key = MAPPING.get(material_name, None)
    if meent_key is None:
        return float(FALLBACK_N.get(material_name, 1.0))

    try:
        from meent.on_numpy.modeler.modeling import read_material_table, find_nk_index
        mat_table = read_material_table()
        n_val = find_nk_index(meent_key, mat_table, wavelength)
        return float(n_val)
    except Exception:
        return float(FALLBACK_N.get(material_name, 1.0))

def build_ucell_from_pattern_torch_with_materials(pattern_np, wavelength,
                                                  ridge_material='Si', groove_material='Air',
                                                  thickness_list=None, n_top=1.45, var_layer=2,
                                                  device='cpu'):
    if thickness_list is None:
        thickness_list = [325] * 8
    n_ridge = resolve_n_for_material(ridge_material, wavelength)
    n_groove = resolve_n_for_material(groove_material, wavelength)
    u = (pattern_np + 1) / 2.0
    ucell = np.ones((len(thickness_list), 1, pattern_np.size), dtype=np.float32) * n_groove
    ucell[0:2, 0, :] = n_top
    ucell[var_layer, 0, :] = u * (n_ridge - n_groove) + n_groove
    # create torch tensor on device and require grad
    t = torch.tensor(ucell, dtype=torch.float32, device=device, requires_grad=True)
    return t, float(n_ridge), float(n_groove)

# ---------------- compute derived values ----------------
if args.n_ridge is None:
    try:
        from meent.on_numpy.modeler.modeling import read_material_table, find_nk_index
        mat_table = read_material_table()
        # try to resolve using ridge_material if user didn't pass numeric n_ridge
        n_ridge = resolve_n_for_material(args.ridge_material, WAVELENGTH)
        log.info(f"Resolved n_ridge from material table or mapping: {n_ridge}")
    except Exception:
        n_ridge = 3.5
else:
    n_ridge = float(args.n_ridge)

n_groove = float(args.n_groove)
DELTA_N_MAG = float(n_ridge - n_groove)
log.info(f"Using Δn magnitude (n_ridge - n_groove) = {DELTA_N_MAG:.6f}")

if PERIOD is not None:
    period = [float(PERIOD)]
else:
    period = [abs(WAVELENGTH / math.sin(math.radians(ANGLE)))]

if args.dump_config:
    log.info("CONFIG SUMMARY:")
    log.info(f" wavelength={WAVELENGTH} nm, period={period[0]} nm, fto={FTO}, ORDER_IDX={ORDER_IDX}")

# ---------------- device & backend selection ----------------
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

use_torch_backend = (device.type == 'cuda')
# allow torch backend on CPU too (but autograd is slower); prefer numpy backend if not cuda and not forcing torch.
backend_id = 2 if use_torch_backend else 0

if args.fd:
    # If user forces FD, prefer numpy backend for consistency/robustness
    backend_id = 0
    use_torch_backend = False

log.info(f"Using device={device} use_torch_backend={use_torch_backend} meent.backend={backend_id}")

# ---------------- graceful shutdown handler ----------------
stop_requested = False
def _on_term(signum, frame):
    global stop_requested
    log.warning("SIGTERM received, will save checkpoint and exit soon.")
    stop_requested = True
signal.signal(signal.SIGTERM, _on_term)

# ---------------- helpers: order selection & eta extraction ----------------
def find_order_index_using_meent(mee_obj, wavelength=WAVELENGTH, angle_deg=ANGLE):
    try:
        if hasattr(mee_obj, 'get_kx_ky_vector'):
            kx_vec, ky_vec = mee_obj.get_kx_ky_vector(wavelength)
            k0 = 2 * math.pi / wavelength
            angles = []
            for kx in np.array(kx_vec).reshape(-1):
                try:
                    angles.append(math.degrees(math.asin(float(kx) / k0)))
                except Exception:
                    angles.append(None)
            diffs = [abs(a - angle_deg) if a is not None else 1e9 for a in angles]
            return int(np.argmin(diffs))
    except Exception:
        pass
    return 1

def extract_eta_from_result_generic(res_obj, order_idx):
    # prefer res_obj.de_ti
    try:
        if hasattr(res_obj, 'de_ti'):
            val = res_obj.de_ti
            if isinstance(val, torch.Tensor):
                arr = val.detach().cpu().numpy()
            else:
                arr = np.array(val)
            arr1 = np.array(arr).squeeze()
            if arr1.ndim == 1 and order_idx < arr1.shape[0]:
                return float(arr1[order_idx])
    except Exception:
        pass
    # fallback: res_te_inc amplitude arrays
    try:
        if hasattr(res_obj, 'res_te_inc'):
            amps = res_obj.res_te_inc
            try:
                amps_np = np.array(amps)
            except Exception:
                amps_np = None
            if amps_np is not None:
                try:
                    amps_sel = amps_np[..., order_idx]
                    return float(np.sum(np.abs(amps_sel)**2))
                except Exception:
                    pass
                try:
                    amps_sel = amps_np[order_idx]
                    return float(np.sum(np.abs(amps_sel)**2))
                except Exception:
                    pass
    except Exception:
        pass
    return 0.0

# ---------------- process one sample: autograd (when Torch backend) ----------------
def process_one_autograd(pattern_np, seed_for_sample=None):
    # build ucell on selected device with requires_grad True
    ucell_t, n_ridge_sample, n_groove_sample = build_ucell_from_pattern_torch_with_materials(
        pattern_np, wavelength=WAVELENGTH,
        ridge_material=args.ridge_material, groove_material=args.groove_material,
        thickness_list=THK, n_top=1.45, var_layer=VAR_LAYER, device=device)

    # call mee with the chosen backend (2 for torch)
    mee = meent.call_mee(backend=backend_id, wavelength=WAVELENGTH, period=period,
                         n_top=1.45, n_bot=1.0, theta=0, phi=0, psi=0,
                         fto=FTO, pol=1, thickness=THK, ucell=ucell_t)

    res = mee.conv_solve()
    if not hasattr(res, 'de_ti'):
        raise RuntimeError("meent result missing de_ti")

    # choose order
    order_idx = ORDER_IDX if ORDER_IDX is not None else None
    if order_idx is None:
        try:
            if isinstance(res.de_ti, torch.Tensor):
                order_idx = int(torch.argmax(res.de_ti).item())
            else:
                order_idx = int(np.argmax(np.array(res.de_ti).squeeze()))
        except Exception:
            order_idx = find_order_index_using_meent(mee, WAVELENGTH, ANGLE)

    # ensure de[order_idx] is a torch tensor connected to graph
    if isinstance(res.de_ti, torch.Tensor):
        eta_tensor = res.de_ti.squeeze()[order_idx]
    else:
        # if de_ti is not torch, we can't backprop through ucell; fall back to FD behavior
        raise RuntimeError("de_ti is not a torch tensor — autograd path requires Torch backend")

    # zero grads and backward
    if ucell_t.grad is not None:
        ucell_t.grad.zero_()
    eta_tensor.backward()

    if ucell_t.grad is None:
        raise RuntimeError("ucell.grad is None after backward — autograd didn't propagate")

    # extract gradient on variable layer -> CPU numpy
    grad_layer = ucell_t.grad[VAR_LAYER, 0, :].detach().cpu().numpy().astype(np.float64)

    # per-cell delta eta approximated as grad * delta_n_cell
    delta_n_per_cell = -pattern_np * DELTA_N_MAG
    delta_eta = grad_layer * delta_n_per_cell

    # normalize
    norm = np.linalg.norm(delta_eta)
    delta_eta_norm = (delta_eta / norm) if norm > 0 else delta_eta.copy()

    X = pattern_np.astype(np.float32).reshape(1,1,-1)
    return X, delta_eta_norm.astype(np.float32), delta_eta.astype(np.float32)

# ---------------- process one sample: finite-difference flips (numpy backend) ----------------
def process_one_fd(pattern_np, seed_for_sample=None):
    ucell0, _, _ = None, None, None
    ucell0 = None
    # build numpy ucell and use numpy meent backend (backend=0)
    # we reuse your earlier numpy builder but inline minimal logic
    try:
        from meent.on_numpy.modeler.modeling import read_material_table, find_nk_index
        # use resolve_n_for_material (above) where possible
    except Exception:
        pass
    # build ucell numpy (simple method)
    # reuse logic from build_ucell_from_pattern_torch_with_materials but in numpy
    u = (pattern_np + 1) / 2.0
    ucell_np = np.ones((len(THK), 1, pattern_np.size), dtype=np.float32) * n_groove
    ucell_np[0:2, 0, :] = 1.45
    ucell_np[VAR_LAYER, 0, :] = u * (n_ridge - n_groove) + n_groove

    mee0 = meent.call_mee(backend=0, wavelength=WAVELENGTH, period=period,
                         n_top=1.45, n_bot=1.0, theta=0, phi=0, psi=0,
                         fto=FTO, pol=1, thickness=THK, ucell=ucell_np)
    res0 = mee0.conv_solve()
    # pick order
    order_idx = ORDER_IDX if ORDER_IDX is not None else None
    if order_idx is None:
        try:
            arr = np.array(res0.de_ti).squeeze()
            order_idx = int(np.argmax(arr))
        except Exception:
            order_idx = find_order_index_using_meent(mee0, WAVELENGTH, ANGLE)

    eta0 = extract_eta_from_result_generic(res0, order_idx)
    M = pattern_np.size
    delta = np.zeros(M, dtype=float)
    for i in range(M):
        p2 = pattern_np.copy()
        p2[i] = -p2[i]
        ucell2 = np.ones_like(ucell_np)
        ucell2[0:2, 0, :] = 1.45
        ucell2[VAR_LAYER, 0, :] = ((p2 + 1) / 2.0) * (n_ridge - n_groove) + n_groove
        mee2 = meent.call_mee(backend=0, wavelength=WAVELENGTH, period=period,
                              n_top=1.45, n_bot=1.0, theta=0, phi=0, psi=0,
                              fto=FTO, pol=1, thickness=THK, ucell=ucell2)
        res2 = mee2.conv_solve()
        eta_i = extract_eta_from_result_generic(res2, order_idx)
        delta[i] = eta_i - eta0

    norm = np.linalg.norm(delta)
    delta_n = (delta / norm) if norm > 0 else delta.copy()
    X = pattern_np.astype(np.float32).reshape(1,1,-1)
    return X, delta_n.astype(np.float32), delta.astype(np.float32)

# ---------------- dataset builder ----------------
def build_dataset(num_samples, prefix='train', seed_offset=0):
    log.info(f"Building dataset {prefix} num_samples={num_samples} seed_offset={seed_offset}")
    rng = np.random.RandomState(SEED_BASE + seed_offset)
    X_list = []
    Y_list = []
    raw_list = []
    saved_paths = []
    for i in range(num_samples):
        if stop_requested:
            log.warning("Stop requested; breaking generation loop to save checkpoint.")
            break
        s = int(SEED_BASE + seed_offset + i)
        # deterministic sample
        rng_local = np.random.RandomState(s)
        pattern = rng_local.choice([1, -1], size=(N,))
        try:
            if use_torch_backend and not args.fd:
                X, Ynorm, raw = process_one_autograd(pattern, seed_for_sample=s)
            else:
                X, Ynorm, raw = process_one_fd(pattern, seed_for_sample=s)
        except Exception as e:
            log.error(f"Sample {i} generation failed: {e}")
            # still record zeros so shapes remain consistent (or skip depending on policy)
            X = pattern.astype(np.float32).reshape(1,1,-1)
            raw = np.zeros(N, dtype=np.float32)
            Ynorm = np.zeros(N, dtype=np.float32)

        X_list.append(X)
        Y_list.append(Ynorm)
        raw_list.append(raw)

        # Checkpoint/save periodically
        if ((i+1) % args.checkpoint_interval == 0) or (i+1 == num_samples):
            X_arr = np.vstack(X_list)   # (k,1,1,N)
            Y_arr = np.vstack(Y_list)   # (k,N)
            raw_arr = np.vstack(raw_list)  # (k,N)
            # ensure X shape is (k,1,N)
            if X_arr.ndim == 4 and X_arr.shape[2] == 1:
                X_arr = X_arr.squeeze(2)
            # metadata
            tnow = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta = {
                'wavelength': WAVELENGTH,
                'angle': ANGLE,
                'N': N,
                'period': period[0],
                'fto': FTO,
                'thickness_layers': len(THK),
                'variable_layer': VAR_LAYER,
                'ridge_material': args.ridge_material,
                'groove_material': args.groove_material,
                'n_ridge': n_ridge,
                'n_groove': n_groove,
                'delta_n_magnitude': DELTA_N_MAG,
                'created': tnow,
                'script': os.path.abspath(__file__),
                'torch_version': getattr(torch, '__version__', None),
                'torch_cuda': getattr(torch.version, 'cuda', None),
                'meent_file': getattr(meent, '__file__', None),
                'seed': SEED_BASE + seed_offset
            }
            chunk_idx = i+1
            suffix = f"{prefix}_adj_N{N}_k{chunk_idx}_{tnow}.pt"
            if args.out_prefix:
                out_name = f"{args.out_prefix}_{suffix}"
            else:
                out_name = suffix
            out_path = os.path.join(OUT_DIR, out_name)
            # types -> torch
            X_t = torch.from_numpy(X_arr).float()
            Y_t = torch.from_numpy(Y_arr).float()
            raw_t = torch.from_numpy(raw_arr).float()
            ds = TensorDataset(X_t, Y_t)
            torch.save({'dataset': ds, 'X': X_t, 'Y': Y_t, 'raw_delta': raw_t, 'meta': meta}, out_path)
            log.info(f"Saved checkpoint: {out_path} (samples {len(X_list)})")
            saved_paths.append(out_path)
            # if chunk-size reached, clear in-memory buffers to save RAM
            if len(X_list) >= args.chunk_size:
                X_list = []
                Y_list = []
                raw_list = []
    return saved_paths

# ---------------- main ----------------
if __name__ == "__main__":
    log.info("Starting generation")
    log.info(f"device={device} backend_id={backend_id} fd_forced={args.fd}")

    # small smoke if asked
    train_paths = build_dataset(TRAIN_SIZE, prefix='train', seed_offset=0)
    test_paths = build_dataset(TEST_SIZE, prefix='test', seed_offset=1_000_000)
    log.info("Finished. train_paths: %s", train_paths)
    log.info("Finished. test_paths: %s", test_paths)
