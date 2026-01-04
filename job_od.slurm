#!/bin/bash
#SBATCH --partition=gpu-teaching-7d   # your long partition
#SBATCH --gpus=1
#SBATCH --exclusive                      # ← NO sharing with other users → fixes node contention
#SBATCH --time=168:00:00                  # 168 hours for 40-epoch training
#SBATCH --job-name=pretrained_swin_object_detection         # job name
#SBATCH --output=logs/%j_%x.out          # includes job name → easier to find
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml17/Machine-Learning-Project
#SBATCH --array=1-1                      # Adjust as needed for multiple runs

mkdir -p logs

set -e  # fail fast if anything goes wrong

# === Best-practice memory fixes (keep all of these!) ===
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2


# === Run your training ===
apptainer run --nv \
    --bind /home/pml17:/home/pml17 \
    od_mmdet.sif \
    python main_od.py
    