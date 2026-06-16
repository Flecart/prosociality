#!/bin/bash
# Smoke-scale spatial-dilemma sweep (A2C + interdependence wrapper) on a compute
# node, so it does not compete with the matrix sweep on the login node.
#   sbatch scripts/slurm/spatial.sbatch
#SBATCH --job-name=pro-spatial
#SBATCH --output=/scratch/schmidt/ssci-michael/%x-%j.out
#SBATCH --error=/scratch/schmidt/ssci-michael/%x-%j.err
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda/13.0.2
export PYTHONPATH=src
srun uv run --project ../constitutional-ai python -m prosocial.experiments.spatial_sweep \
    --seeds 3 --episodes 150 --k-alpha 4 --workers 6 "$@"
