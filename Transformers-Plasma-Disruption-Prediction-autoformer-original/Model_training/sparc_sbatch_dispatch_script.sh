#!/bin/bash
#SBATCH --job-name=disruption_wandb_sweep
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --time=7:00:00
#SBATCH -p sched_mit_psfc_gpu_r8

source /etc/profile

# Load necessary modules
module load anaconda3/2022.10
module load cuda/10.2
module load cudnn/8.2.2_cuda10.2

# load lucas' wandb_api. This should change for other users...
# export WANDB_API_KEY=$(cat lucas_wandb_api)

# Activate your virtual environment
source /home/software/anaconda3/2022.10/etc/profile.d/conda.sh
conda activate py39

wandb login $(cat lucas_wandb_api)

# Run multiple instantiations of the WandB agent in parallel
wandb agent "$1"  

# Deactivate your virtual environment
conda deactivate
