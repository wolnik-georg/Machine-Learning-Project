#!/bin/bash
#SBATCH --job-name=cifar100_linear_probe
#SBATCH --partition=gpu-teaching-5h  # Use 5h partition for full training
#SBATCH --gpus=1
#SBATCH --account=pml20
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs
apptainer run --nv pml.sif python main.py "$@"
