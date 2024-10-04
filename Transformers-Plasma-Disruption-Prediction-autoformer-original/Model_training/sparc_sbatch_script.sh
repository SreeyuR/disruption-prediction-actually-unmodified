#!/bin/bash
#SBATCH --job-name=disruption_wandb_sweep
#SBATCH -N 8
#SBATCH --gres=gpu
#SBATCH --time=36:00:00
#SBATCH -p sched_mit_psfc_r8

# Load necessary modules
module load anaconda3/2022.10
module load cuda/9.0
module load cudnn/8.2.2_cuda10.2

# Activate your virtual environment
source activate py39

# Run multiple instantiations of the WandB agent in parallel
for i in {1.."$1"}
do
  wandb agent "$2" & 
done

# Wait for all agent instances to finish
wait

# Deactivate your virtual environment
conda deactivate
