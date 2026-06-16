# Code guide — what to read, in what order

This project has **two separate paradigms**. The confusion about "A2C/IQL vs
training LLMs" comes from mixing them up — they are different tracks.

## TL;DR: why A2C and IQL (not GRPO/DPO) for most of the project

The paper's core experiments are **multi-agent reinforcement learning in social
dilemmas**. The "agents" playing Prisoner's Dilemma / Public Goods / Coin Game
are **not LLMs** — each is a tiny RL policy:
- a **Q-table** trained with **tabular independent Q-learning (IQL)** for the
  matrix games, and
- a **small MLP** trained with **advantage actor-critic (A2C)** for the
  gridworlds.

IQL and A2C are the *standard environment-RL algorithms* for these benchmarks
(Leibo et al. 2017). They take actions in an environment with states/transitions
and per-step payoffs. **GRPO and DPO are LLM-policy-optimization algorithms** and
do not apply to a Q-table or a gridworld MLP — they live only in the separate LLM
track below.

The single shared object across both tracks is the interdependence reward
transform `U = (I-A)^{-1} π`.

---

## Track A — Multi-agent RL (the IQL / A2C track, ~80% of the paper)

Read in this order:

1. **`src/prosocial/interdependence.py`** — the heart: `U = (I-A)^{-1}π`,
   feasibility `ρ(A)<1`, the 2-player closed form. ~40 lines, pure numpy.
2. **`src/prosocial/rewards.py`** — reward wrappers that sit between the env and
   the learner: `Selfish`, `Interdependence` (symmetric), `RewardShaping`,
   `GraphInterdependence` (arbitrary A), `NeighborShaping`. This is the "modular
   reward wrapper."
3. **`src/prosocial/envs/matrix.py`** — IPD, Stag Hunt, Public Goods + the
   repeated-game wrapper (horizons).
4. **`src/prosocial/envs/spatial.py`** — Coin Game, Harvest, Cleanup gridworlds
   (self-contained, no Melting-Pot dependency).
5. **`src/prosocial/agents/qlearning.py`** — tabular IQL (the matrix-game learner;
   ε-greedy + Boltzmann). **This is one of the "agents," not an LLM.**
6. **`src/prosocial/agents/a2c.py`** — independent A2C actor-critic (the gridworld
   learner; a 2-layer MLP policy+value). **Also not an LLM.**
7. **`src/prosocial/train.py`** — matrix self-play loop: env → raw π →
   `transform(π)` → IQL update. Metrics on raw π.
8. **`src/prosocial/train_spatial.py`** — same, for gridworlds with A2C.

Experiments (each is a small driver that sweeps a parameter and writes JSONL):
- `experiments/phase_transition.py` — the headline α-sweep (1,280 runs).
- `experiments/learner_robust.py` — transition under 3 IQL variants + (folds in)
  A2C, to show it's not an exploration artifact.
- `experiments/group_size.py` — feasibility law across N.
- `experiments/graph_structure.py`, `experiments/topology.py` — structure vs
  shaping on non-complete graphs.
- `experiments/spatial_sweep.py` — the spatial runs.
- `experiments/a2c_matrix.py` — A2C on the one-shot IPD (on-policy confirmation).

Plotting: `src/prosocial/plotting*.py` (one per experiment) → `outputs/figures/`.

---

## Track B — LLMs (the GRPO / prompting track)

This is where GRPO lives. **No A2C/IQL here.**

1. **`scripts/llm_game_eval.py`** — the in-context **probe**: render a dilemma as
   text, put the utility transform in the system prompt, sample the model's
   action. *No training* — this is what the paper's LLM figure currently uses.
2. **`scripts/smoke_grpo.py`** — the original **GRPO + LoRA smoke** (Qwen-0.6B,
   4 steps) that validated the TRL training stack; reward = interdependence-
   transformed PD payoff.
3. **`scripts/grpo_train.py`** — **full GRPO + LoRA training** (NEW): trains a
   model so its policy optimizes `U_i`, logging the cooperation rate every N
   steps; runs at a caring α=0.8 and a selfish α=0 control. This is the proper
   "train the LLM agents" experiment. Launched for all 4 models × 2 α.

GRPO uses TRL (`GRPOTrainer`) from the sibling `../constitutional-ai` environment
(which is itself a GRPO/DPO scaffold). To run on LLMs you sample completions and
score them with a reward function — that's why GRPO fits LLMs and A2C/IQL fit the
gridworld agents.

---

## SLURM
`scripts/slurm/` — launchers. Matrix/spatial/experiments run on CPU nodes
(`matrix_sweep.sbatch`, `experiment.sbatch`); LLM eval + GRPO training run on the
`b200` GPU partition (`llm_eval.sbatch`, `grpo_train.sbatch`). Logs land in
`/scratch/schmidt/ssci-michael/`.

## Paper
`paper/content.tex` (manuscript), `paper/reviews/` (4 reviewer rounds + external
panels + rebuttals), `paper/provenance/` (LLM checkpoint hashes + run logs).

## One-line mental model
> **Track A** trains tiny RL agents (Q-tables / MLPs) to *play* social dilemmas,
> using IQL/A2C, to study the interdependence reward. **Track B** prompts and
> (now) GRPO-trains *LLMs* on the same reward. The transform `(I-A)^{-1}π` is the
> bridge.
