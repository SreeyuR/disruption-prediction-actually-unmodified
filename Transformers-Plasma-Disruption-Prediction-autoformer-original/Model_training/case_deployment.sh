#!/bin/bash

#SBATCH --job-name=sweeps
#SBATCH --output=sweeps_%j.out
#SBATCH --error=sweeps_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --array=0-35
#SBATCH --partition=sched_mit_psfc_r8

source /etc/profile

# Load necessary modules
module load anaconda3/2022.10
module load cuda/10.2
module load cudnn/8.2.2_cuda10.2

# Activate your virtual environment
source /home/software/anaconda3/2022.10/etc/profile.d/conda.sh
conda activate py39

# Define the possible values for C and D
c_values=("east" "d3d" "cmod")
d_values=(1 2 3 4 5 6 7 8 9 10 11 12)

# Initialize an empty SWEEP_CONFIGS array
SWEEP_CONFIGS=()

# Iterate through the possible values of C and D, and create the sweep configuration filenames
for c_value in "${c_values[@]}"; do
    for d_value in "${d_values[@]}"; do
        sweep_config_file="sweep_configs/sweep_config_${c_value}_${d_value}.yaml"
        SWEEP_CONFIGS+=("$sweep_config_file")
    done
done

# Get the sweep configuration for this task
SWEEP_CONFIG_FILE=${SWEEP_CONFIGS[${SLURM_ARRAY_TASK_ID}]}

python3 case_deployment_helper.py $SWEEP_CONFIG_FILE
