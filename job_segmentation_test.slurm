#!/bin/bash
#SBATCH --partition=gpu-teaching-9m
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:09:00
#SBATCH --job-name=resnet101_test
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --chdir=/home/pml20/Machine-Learning-Project

# =============================================================================
# Quick Test: ResNet-101 + UperNet (9 minutes)
# =============================================================================
# Just to verify the pipeline works before the full 7-day run starts
# =============================================================================

mkdir -p logs

echo "========================================="
echo "ResNet-101 + UperNet Quick Test"
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "========================================="

# GPU info
nvidia-smi

# Set PyTorch CUDA memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# Run segmentation training (will only run for ~8 minutes before timeout)
apptainer run --nv \
    pml.sif \
    python main_segmentation.py

echo "Test completed at: $(date)"
