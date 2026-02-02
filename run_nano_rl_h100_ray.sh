#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=nano_rl_logs/nano_rl_%A.out
#SBATCH --error=nano_rl_logs/nano_rl_%A.err
#SBATCH --gres=gpu:h100:2
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=17
#SBATCH --account=dtce-schmidt
#SBATCH --partition=short
#SBATCH --job-name=nano_rl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mengyun.wang@eng.ox.ac.uk

set -e
set -x


# --- environment setup ---
module load Anaconda3

# Use conda recommended activation for scripts
eval "$(conda shell.bash hook)"

# provide a safe default so qt activation scripts don't fail
# (the script expects this variable to be set; empty string is safe)
export QT_XCB_GL_INTEGRATION=""

# activate the env (adjust if $DATA path differs)
conda activate "$DATA/venvs/nano_rl"

echo "Job start: $(date) on $(hostname)"
echo "User: $(whoami)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_GPUS: ${SLURM_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_PER_TASK}"

which python
python -c "import sys, torch, ray; print('python', sys.executable); print('torch.cuda.is_available()', torch.cuda.is_available()); print('torch.cuda.device_count()', getattr(torch.cuda,'device_count', lambda:0)()); print('ray', ray.__version__)"
nvidia-smi

# raise ulimits
ulimit -n 65535 || true
ulimit -u 65535 || true

# go to repo
cd $DATA/repos/Physics-Informed-Reinforcement-Learning

# ensure log dir exists
mkdir -p nano_rl_logs
mkdir -p run
mkdir -p ray_saved_sessions

# helper: compute number of visible GPUs
GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # CUDA_VISIBLE_DEVICES like "0,1" or "0"
  GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
  # fallback to nvidia-smi query
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
fi
echo "Detected GPU_COUNT=$GPU_COUNT"

# start ray head on compute node so local workers can register
ray stop || true

NODE_IP=$(hostname -I | awk '{print $1}')
echo "Starting ray head on $NODE_IP (num-cpus=$SLURM_CPUS_PER_TASK num-gpus=$GPU_COUNT)"
# if GPU_COUNT is 0, still start ray but advertise 0 gpus
if [[ "$GPU_COUNT" -gt 0 ]]; then
  ray start --head --node-ip-address="$NODE_IP" --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPU_COUNT" --include-dashboard=false
else
  ray start --head --node-ip-address="$NODE_IP" --num-cpus="$SLURM_CPUS_PER_TASK" --include-dashboard=false
fi

# ensure ray session logs are copied back when the job exits
trap 'echo "Job ending: copying ray session logs..."; rsync -a --prune-empty-dirs --relative /tmp/ray/session_* '"$PWD"'/ray_saved_sessions/ || true; ray stop || true; echo "done."' EXIT

# run training
python main_ray_slurm.py --data_dir $DATA/repos/Physics-Informed-Reinforcement-Learning/run

# ray stop will be called by the trap on exit

