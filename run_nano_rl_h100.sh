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
module load Anaconda3
source activate base
conda activate $DATA/venvs/nano_rl
which python
nvidia-smi
ulimit -n 65535
ulimit -u 65535
cd $DATA/repos/Physics-Informed-Reinforcement-Learning
python main.py --data_dir $DATA/repos/Physics-Informed-Reinforcement-Learning/run

