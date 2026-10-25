#!/bin/bash
# =====================================================================
#  GENERIC job without GPU (data preparation, downloads). Usage:
#     sbatch -J job_name jobs/cpu.sbatch  script.py  --arguments
#  Example:
#     sbatch -J prep jobs/cpu.sbatch 01_prepare_data.py
# =====================================================================
#SBATCH --job-name=splade_cpu
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lce
cd "$SLURM_SUBMIT_DIR"
set -e

echo "Node: $(hostname) | Job: $SLURM_JOB_ID | Start: $(date)"
echo "WORK_DIR=${WORK_DIR:-(config.py default)}"

python -u "$@"

echo "End: $(date)"
