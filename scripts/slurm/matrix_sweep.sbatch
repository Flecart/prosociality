#!/bin/bash
# Phase-transition matrix sweep on a DEDICATED compute node (the login node is
# CPU-contended by other users, which starves the pool). CPU-only job; the GPU
# request just secures an allocation on the b200 partition.
#   sbatch scripts/slurm/matrix_sweep.sbatch
#SBATCH --job-name=pro-matrix
#SBATCH --output=/scratch/schmidt/ssci-michael/%x-%j.out
#SBATCH --error=/scratch/schmidt/ssci-michael/%x-%j.err
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH=src
srun uv run --project ../constitutional-ai python -m prosocial.experiments.phase_transition \
    --seeds 10 --episodes 3000 --workers 15 --out results/phase_transition.jsonl "$@"
