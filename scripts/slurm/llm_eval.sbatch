#!/bin/bash
# LLM agent-behavior eval on matrix social dilemmas (Qwen + Gemma families).
#   sbatch scripts/slurm/llm_eval.sbatch
#SBATCH --job-name=pro-llm-eval
#SBATCH --output=/scratch/schmidt/ssci-michael/%x-%j.out
#SBATCH --error=/scratch/schmidt/ssci-michael/%x-%j.err
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda/13.0.2
export HF_HUB_ENABLE_HF_TRANSFER=1

# Qwen3-0.6B (cached) is a fast sanity model so we always get a real curve even
# if the large/gated downloads fail; harness records unavailable models cleanly.
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3.6-35B-A3B google/gemma-3-12b-it google/gemma-4-12B}"
srun uv run --project ../constitutional-ai python scripts/llm_game_eval.py \
    --models $MODELS \
    --alphas 0 0.3 0.6 0.9 --samples 16 \
    --out results/llm_games.jsonl "$@"
