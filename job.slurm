#!/bin/bash
#SBATCH --partition=gpu-teaching-2h  # or gpu for longer
#SBATCH --gpus=1
#SBATCH --time=08:00:00  # 8 hours for 30-epoch ablation training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ML_Train
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project

mkdir -p logs
apptainer run --nv --overlay /home/space/datasets-sqfs/imagenet2012.sqfs pml.sif python main.py
