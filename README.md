# Prosociality — Structural Interdependence for Multi-Agent Cooperation

Implementation + paper for *"Beyond the Folk Theorem: Structural Interdependence
in Reward Functions as a Mechanism for Multi-Agent Cooperation."* We import the
Bergstrom (1999) benevolence system `U = (I-A)^{-1} π` into multi-agent RL as a
modular reward wrapper and characterize it empirically.

## Layout
- `src/prosocial/` — platform: interdependence transform, reward families, matrix
  + spatial environments, IQL/A2C learners, experiments, plotting.
- `scripts/slurm/` — SLURM launchers (matrix sweep, spatial, LLM eval, generic).
- `results/` — JSONL outputs; `outputs/figures/` — generated figures.
- `paper/` — the manuscript (own git repo); `paper/reviews/` — reviewer rounds +
  rebuttals; `paper/provenance/` — LLM checkpoint hashes + run logs.
- `REPORT.md` — standalone experiment report with embedded plots.

## What was done
- **Platform + 7 cases**: IPD, Stag Hunt, Public Goods (matrix, one-shot +
  repeated H∈{1,5,10,100}); Coin Game, Harvest, Cleanup (spatial gridworlds).
- **Experiments**: one-shot cooperation phase transition (1,280 runs); learner
  robustness across 4 learners incl. on-policy A2C; group-size feasibility law
  α<1/(N−1) across N; structure-vs-shaping (chain + topology generalization with
  CIs); LLM behavior for 4 models (Qwen 0.6B / 35B, Gemma-3-12B, Gemma-4-12B).
- **Review**: 4 iterated ICML rounds + a 5-agent panel + PaperMentor writing
  review + AI-Scientist NeurIPS reviewer; iterated to a re-review verdict of
  Weak Accept (committed per round in `paper/`).

## Run
```bash
PYTHONPATH=src uv run --project ../constitutional-ai python -m prosocial.experiments.phase_transition --seeds 10 --episodes 3000 --workers 14
sbatch scripts/slurm/matrix_sweep.sbatch     # on a dedicated compute node
sbatch scripts/slurm/llm_eval.sbatch         # LLM behavior (b200)
PYTHONPATH=src uv run --project ../constitutional-ai --with matplotlib python -m prosocial.plotting
```
Note: the HF cache lives on scratch (`/scratch/schmidt/ssci-michael/hf_hub_huangelo`,
symlinked from `~/.cache/huggingface/hub`) because the home quota cannot hold the
67G Qwen-35B alongside other downloads.

---

ended at 01:35 BST on Tuesday 16 June 2026
