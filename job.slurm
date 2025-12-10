#!/bin/bash
#SBATCH --partition=gpu-teaching-7d  # Longer partition for 100 epochs
#SBATCH --gpus=1
#SBATCH --time=168:00:00  # 7 days for 100 epochs
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=imagenet_100_epochs
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project

mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
apptainer run --nv --overlay /home/space/datasets-sqfs/imagenet2012.sqfs pml.sif python main.py