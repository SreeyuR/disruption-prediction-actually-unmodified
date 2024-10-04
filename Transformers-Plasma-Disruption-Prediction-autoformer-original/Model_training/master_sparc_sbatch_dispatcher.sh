#!/bin/bash

for i in $(seq 1 "$1")
do
  if [ "$3" == "gpu" ]; then
    sbatch sparc_sbatch_dispatch_script.sh "$2" &
  elif [ "$3" == "cpu" ]; then
    sbatch sparc_sbatch_dispatch_cpu_script.sh "$2" &
  else
    echo "Invalid argument for GPU/CPU"
  fi
done