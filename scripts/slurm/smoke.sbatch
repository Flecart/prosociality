#!/bin/bash
# Tiny GRPO+LoRA smoke for the prosociality stack on one B200.
#   sbatch scripts/slurm/smoke.sbatch
# Reuses the constitutional-ai uv environment (trl/peft/torch already there).
#SBATCH --job-name=pro-smoke
#SBATCH --output=/scratch/schmidt/ssci-michael/%x-%j.out
#SBATCH --error=/scratch/schmidt/ssci-michael/%x-%j.err
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda/13.0.2
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run with the existing constitutional-ai venv (has trl/peft/qwen cached).
ENV_PROJECT="${ENV_PROJECT:-../constitutional-ai}"
srun uv run --project "$ENV_PROJECT" python scripts/smoke_grpo.py "$@"
