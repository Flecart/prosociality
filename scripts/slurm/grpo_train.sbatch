#!/bin/bash
# Full GRPO+LoRA training of an LLM agent on the interdependence-transformed PD.
#   MODEL=Qwen/Qwen3-0.6B ALPHA=0.8 sbatch scripts/slurm/grpo_train.sbatch
#SBATCH --job-name=pro-grpo
#SBATCH --output=/scratch/schmidt/ssci-michael/%x-%j.out
#SBATCH --error=/scratch/schmidt/ssci-michael/%x-%j.err
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda/13.0.2
export HF_HUB_ENABLE_HF_TRANSFER=1
MODEL="${MODEL:?set MODEL}"
ALPHA="${ALPHA:-0.8}"
STEPS="${STEPS:-200}"
srun uv run --project ../constitutional-ai python scripts/grpo_train.py \
    --model "$MODEL" --alpha "$ALPHA" --steps "$STEPS" "$@"
