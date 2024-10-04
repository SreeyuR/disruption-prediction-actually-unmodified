import sys
import wandb
import os
import yaml

def create_sweep(sweep_config, project_name):
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    return sweep_id

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 create_and_run_sweep.py SWEEP_CONFIG_FILE")
        sys.exit(1)

    sweep_config_file = sys.argv[1]
    project_name = "HDL-improvement-transformer"

    with open(sweep_config_file, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = create_sweep(sweep_config, project_name)
    wandb.agent(sweep_id)
