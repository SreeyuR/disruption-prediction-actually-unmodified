#!/bin/bash
#SBATCH --job-name=disruption_wandb_sweep
#SBATCH -N 1
#SBATCH --gpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH -p regular 

# Run multiple instantiations of the WandB agent in parallel

source model_env_gpu/bin/activate

wandb agent "$1"  

# Wait for all agent instances to finish
wait
