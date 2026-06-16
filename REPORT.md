# Prosociality: Structural Interdependence for Multi-Agent Cooperation — Experiment Report

**Thesis.** Cooperation in multi-agent systems is usually engineered *temporally*
(repeated play + punishment, the Folk Theorem). We study a complementary axis:
**structural interdependence** in the reward function, where each agent's
effective utility is coupled to others' via the Bergstrom (1999) transform
`U = (I - A)^{-1} π`, well-posed when the spectral radius `ρ(A) < 1`. We
implement this as a modular reward wrapper and benchmark it across matrix and
spatial social dilemmas, plus an in-context LLM probe.

This report summarizes the runnable platform and the empirical findings. The
paper draft is in `paper/` (`content.tex`); figures are in `outputs/figures/`.

## Platform (`src/prosocial/`)

| Component | Module |
|---|---|
| Core transform `U=(I-A)^{-1}π` + feasibility `ρ(A)<1` | `interdependence.py` |
| Reward families: selfish / interdependence / shaping | `rewards.py` |
| Matrix games: IPD, Stag Hunt, Public Goods + repeated wrapper (H∈{1,5,10,50,100}) | `envs/matrix.py` |
| Spatial dilemmas: Coin Game, Harvest, Cleanup (self-contained gridworlds) | `envs/spatial.py` |
| Learners: tabular independent Q-learning; independent A2C | `agents/` |
| Self-play training loops | `train.py`, `train_spatial.py` |
| Experiments + plotting | `experiments/`, `plotting*.py` |

All matrix experiments are CPU-bound; spatial uses small MLPs (CPU/GPU). SLURM
launchers in `scripts/slurm/`.

## Experiment 1 — Cooperation phase transition (matrix games)

Sweep of interdependence `α` × horizon × game × 10 seeds (1,380 runs), cooperation
measured on **raw** payoffs over the final 10% of training.

![Phase transition](outputs/figures/phase_transition_coop.png)

**Findings.**
- **One-shot cooperation emerges** in the Prisoner's Dilemma as `α` crosses a
  critical region near the theoretical `α*=2/3` — with *no* repetition, where the
  Folk Theorem cannot apply.
- **Stag Hunt** shows the sharpest transition: interdependence flips the selected
  equilibrium from risk-dominant Hare to payoff-dominant Stag.
- **Group-size feasibility law:** for symmetric coupling `ρ(A)=(N−1)α`, so
  `α < 1/(N−1)`. The 4-player Public Goods Game only cooperates as `α` approaches
  its ceiling `1/3` — feasible interdependence shrinks with group size, a
  constraint with no temporal analogue.

![Interdependence vs shaping](outputs/figures/interdep_vs_shaping.png)

Structural interdependence and flat reward shaping `r_i=π_i+β Σπ_j` both lift
one-shot cooperation; the structural difference is expected to manifest in the
*dynamic* settings (exploitation, endogenous α, cascades) listed in Future Work.

![IPD welfare](outputs/figures/ipd_welfare.png)

## LLM behavior — in-context interdependence

Each one-shot game is rendered as text with the utility transform in the system
prompt; we sample the action 16× per (model, game, α).

![LLM cooperation](outputs/figures/llm_coop_vs_alpha.png)

**Four distinct behavioral signatures:**
- **Qwen3.6-35B-A3B** — *graded* internalization: cooperation rises monotonically
  with `α` (PD 0.00→0.88, PGG 0.00→0.75). Closest in-context analogue of the RL
  transition.
- **gemma-3-12b-it** — *threshold switcher*: 0→1.0 jump at `α≥0.3` in Stag Hunt
  and PGG, but **stubbornly defects in PD at every α**.
- **gemma-4-12B** — *noisy/partial*: non-monotone, lower valid-parse rates (verbose).
- **Qwen3-0.6B** — *saturated*: trivially cooperates everywhere (too small to
  engage the trade-off).

"Prompted prosociality" is not a single capability but a model-specific behavior.

## Spatial dilemmas (smoke scale)

![Spatial](outputs/figures/spatial_coop_welfare.png)

The same wrapper composes with A2C on gridworlds. Coin Game shows the clearest
structural effect (shift away from destructive coin-stealing); Harvest/Cleanup
need longer training. These validate plumbing, not benchmark-scale claims.

## Reproduce

```bash
# matrix phase transition (CPU, ~10-20 min, 14 workers)
PYTHONPATH=src uv run --project ../constitutional-ai python -m prosocial.experiments.phase_transition --seeds 10 --episodes 4000 --workers 14
# spatial sweep (SLURM)
sbatch scripts/slurm/spatial.sbatch
# LLM behavior (SLURM, b200)
sbatch scripts/slurm/llm_eval.sbatch
# figures
PYTHONPATH=src uv run --project ../constitutional-ai --with matplotlib python -m prosocial.plotting
```

> Note: the HuggingFace cache lives on scratch (`/scratch/schmidt/ssci-michael/hf_hub_huangelo`,
> symlinked from `~/.cache/huggingface/hub`) because the home quota (~94G) cannot
> hold the 67G Qwen-35B alongside other downloads.
