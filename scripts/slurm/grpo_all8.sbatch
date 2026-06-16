#!/bin/bash
# All 8 GRPO trainings (4 models x {caring alpha=0.8, selfish alpha=0}) on ONE
# exclusive node, one training per GPU. A single allocation -> not preempted
# piecemeal, and clears the queue in one shot.
#   sbatch scripts/slurm/grpo_all8.sbatch
#SBATCH --job-name=pro-grpo8
#SBATCH --output=/scratch/schmidt/ssci-michael/%x-%j.out
#SBATCH --error=/scratch/schmidt/ssci-michael/%x-%j.err
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=6:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda/13.0.2
export HF_HUB_ENABLE_HF_TRANSFER=1

MODELS=("Qwen/Qwen3-0.6B"       "Qwen/Qwen3-0.6B" \
        "Qwen/Qwen3.6-35B-A3B"  "Qwen/Qwen3.6-35B-A3B" \
        "google/gemma-3-12b-it" "google/gemma-3-12b-it" \
        "google/gemma-4-12B"    "google/gemma-4-12B")
ALPHAS=(0.8 0.0 0.8 0.0 0.8 0.0 0.8 0.0)

pids=()
for i in 0 1 2 3 4 5 6 7; do
  echo "[launch] GPU $i: ${MODELS[$i]} alpha=${ALPHAS[$i]}"
  CUDA_VISIBLE_DEVICES=$i uv run --project ../constitutional-ai \
      python scripts/grpo_train.py --model "${MODELS[$i]}" --alpha "${ALPHAS[$i]}" \
      --steps 150 > "/scratch/schmidt/ssci-michael/grpo8_gpu${i}.log" 2>&1 &
  pids+=($!)
  sleep 45   # stagger model loads so 8 simultaneous loads don't spike host RAM
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[done] GPU $i OK"; else echo "[done] GPU $i FAILED"; fail=1; fi
done
echo "ALL 8 GRPO TRAININGS COMPLETE (fail=$fail)"
