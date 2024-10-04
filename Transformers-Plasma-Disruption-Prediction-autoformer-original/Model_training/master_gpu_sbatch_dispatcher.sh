#!/bin/bash

for i in {1.."$1"}
do
  sbatch gpu_sbatch_script.sh "$2" & 
done