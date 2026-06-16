# SLURM Playbook

How to interact with this SLURM cluster. Distilled from the `constitutional-ai`
repo's run scripts; the workload commands are CAI-specific but the cluster
conventions below are general.

## The cluster
- **Partition:** `b200` — NVIDIA Blackwell GPUs (sm_100). The only GPU partition used.
- **CUDA:** `module load cuda/13.0.2` at the top of every GPU job. Torch must come
  from the **cu128** index (Blackwell). 
- **Python/env:** system Python is too old — everything runs through **uv**
  (`uv run ...`, `uv sync`). Login node has internet (do `uv sync` / model
  downloads there). Compute nodes also have internet (external API calls work).
- **Scratch / logs:** job stdout/stderr go to
  `/scratch/schmidt/ssci-michael/%x-%j.{out,err}` (`%x`=job name, `%j`=job id).

## Two ways to get a GPU

### Interactive (quick tests, debugging)
```bash
srun --partition=b200 --gres=gpu:1 --cpus-per-task=8 --time=1:00:00 --pty bash
# then inside: module load cuda/13.0.2 ; uv run <cmd>
```

### Batch (sbatch) — standard skeleton
```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --output=/scratch/schmidt/ssci-michael/%x-%j.out
#SBATCH --error=/scratch/schmidt/ssci-michael/%x-%j.err
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=<H>:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda/13.0.2
export HF_HUB_ENABLE_HF_TRANSFER=1
srun uv run <command> "$@"
```

Conventions worth copying:
- **1 node, 1 GPU, 8 CPUs** is the standard shape for every job here.
- Wrap the real workload in `srun uv run ...` (runs as a proper step).
- Forward `"$@"` so you can append overrides on the `sbatch` line.
- `export HF_HUB_ENABLE_HF_TRANSFER=1` for fast model downloads.
- Size `--time` per stage (e.g. smoke 1h, light stages 4h, training 8–12h).

## Monitoring
```bash
squeue -u "$USER"          # your jobs
sinfo -p b200              # partition state
tail -f /scratch/schmidt/ssci-michael/<name>-*.out
scancel <jobid>            # cancel a job
```

## Job dependency chains (multi-stage pipelines)
Orchestrate pipelines as dependent jobs rather than one long job:
```bash
jid() { sbatch --parsable "$@"; }                 # capture just the job id
prep=$(jid  --export=ALL prep.sbatch  "$arg")
train=$(jid --export=ALL --dependency=afterok:$prep  train.sbatch "$arg")
eval=$(jid  --export=ALL --dependency=afterok:$train eval.sbatch  "$arg")
```
- `--parsable` → returns just the numeric job id for capturing.
- `--dependency=afterok:<id>` → start only if the prior job succeeded.
- `--export=ALL` → pass the submitting environment through.
- Read positional args inside the script via `${1:?usage...}` and forward them
  to the workload as needed.
- Make scripts **resumable**: check for existing output artifacts and skip
  completed stages.

## Takeaway template
For any new GPU job: copy an existing `.sbatch`, keep the `b200 / 1 node / 1 GPU
/ 8 CPU` shape, log to `/scratch/schmidt/ssci-michael/`, `module load
cuda/13.0.2`, run the real command under `srun uv run ...`, and forward `"$@"`.
For pipelines, submit with `--parsable` + `--dependency=afterok:` chains via a
small driver script.
