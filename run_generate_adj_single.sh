#!/usr/bin/env bash
#SBATCH --job-name=gen-adj-single
#SBATCH --output=logs/gen_adj_single_%j.out
#SBATCH --error=logs/gen_adj_single_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1            # adjust GPU type (h100/v100/a100) to the partition you request
#SBATCH --time=12:00:00             # adjust wallclock as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # number of CPU workers for Python multiprocessing
#SBATCH --mem=64G                   # adjust memory
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mengyun.wang@ox.ac.uk

# -------------------------
# Activate conda env (robust)
# -------------------------
# method A (preferred if conda is available via module)
module load Anaconda3 || true
# ensure 'conda' command is available in non-interactive shells
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc" || true
fi
# fallback: try direct activate if you have env path
CONDA_ENV="/data/dtce-schmidt/oums1256/venvs/meent_test"
# Try conda activate
if command -v conda >/dev/null 2>&1; then
  # make conda work in non-interactive job shells
  . "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  # fallback (works if env is directory created with `conda create -p /path`)
  source "${CONDA_ENV}/bin/activate"
fi

# confirm python / torch
python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", getattr(torch, "__version__", "no torch"))
print("cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
PY

# -------------------------
# Run dataset generator
# -------------------------
REPO="$DATA/repos/Physics-Informed-Reinforcement-Learning"
cd "${REPO}" || exit 2

OUTDIR="$REPO/adj_N256"
mkdir -p "${OUTDIR}"

# Example invocation (edit parameters as you like)
python tools/generate_adj_dataset_arc.py \
  --N 256 \
  --train_size 80 \
  --test_size 20 \
  --out_dir "${OUTDIR}" \
  --wavelength 1100 \
  --angle 60 \
  --workers 8 \
  --checkpoint_interval 100 \
  --fto 40

# NOTE: --workers should not exceed --cpus-per-task
