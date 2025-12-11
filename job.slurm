#!/bin/bash
#SBATCH --partition=gpu-teaching-5h # 2 day partition
#SBATCH --gpus=1
#SBATCH --time=05:00:00  # 48 hours for training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=Ticket10_rel_and_abs_bias
#SBATCH --chdir=/home/pml04/swin_transformer/Machine-Learning-Project

mkdir -p logs

# Memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# Clear GPU memory and cache at start
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()"

apptainer run --nv --overlay /home/space/datasets-sqfs/imagenet2012.sqfs pml.sif python main.py
