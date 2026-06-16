#!/bin/bash
# Generic CPU experiment runner on a dedicated compute node (login node is
# CPU-contended). Pass the module and its args, e.g.:
#   sbatch scripts/slurm/experiment.sbatch prosocial.experiments.learner_robust --seeds 10
#SBATCH --job-name=pro-exp
#SBATCH --output=/scratch/schmidt/ssci-michael/%x-%j.out
#SBATCH --error=/scratch/schmidt/ssci-michael/%x-%j.err
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0:40:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH=src
MODULE="${1:?usage: experiment.sbatch <module> [args...]}"; shift
srun uv run --project ../constitutional-ai python -m "$MODULE" "$@"
