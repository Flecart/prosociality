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

---

# Part II — Collaboration-Based Endogenous Interdependence (new study)

**Question.** The fixed-`A` study above leaves open *where the relational matrix
comes from*. Inspired by the ethological **interdependence hypothesis**
(Tomasello et al. 2012; Roberts 2005), we make `A` an **endogenous, behavioral
readout of observed cooperation**: agents come to care about the specific
partners they have successfully cooperated with. This is distinct from the
gradient endogenization in `experiments/endogenous.py` (which ascends each
agent's *own* payoff and collapses to asymmetric carer/free-rider). The new
mechanism is *not a gradient* — `A_ij` is a saturating function of the running
tally `C_ij` of **joint cooperative acts** (joint log-lifts; joint successful
stag hunts).

Paper: `paper/content_new.tex` (+ `paper/acl_new.tex`, compiles to 8pp).
`paper/content.tex` is left untouched.

## New platform pieces
| Component | Module |
|---|---|
| Collaboration matrix `C -> A` (symmetric, EMA-decayed, `safe_A` clip) | `src/prosocial/collaboration.py` |
| Scale-normalized transform (row-normalized `(I-A)^{-1}`) | `interdependence.normalized_effective_utilities` (+ `normalize=` flag on `rewards.Interdependence`/`GraphInterdependence`) |
| N-player threshold stag hunt (normal-form log-hunt) | `envs/matrix.StagHuntN` |
| Spatial log-hunt (clean/harvest/**lift**, ≥2 co-lifters, records who co-lifted) | `envs/spatial.CleanupStag` |
| PPO learner (clipped, on-policy) | `agents/ppo.PPOLearner` |
| Matrix experiment (bootstrap / groupsize / freerider / trajectory) | `experiments/collab_matrix.py` |
| 4-learner comparison (IQL/DQN/A2C/PPO) | `experiments/collab_algos.py` |
| Spatial experiment (episode-updated A threaded into A2C self-play) | `experiments/collab_spatial.py` |
| Figures | `plotting_collab.py` -> `paper/figures/collab_*.png` |
| Tests | `tests/test_cleanup_stag.py`, `tests/test_collaboration.py` |

## M0 — reproduction (no-logs Cleanup)
Local venv (`uv venv .venv`; numpy + CPU torch). Reproduced `spatial_smoke.jsonl`
Cleanup rows at smoke scale (4 α, 3 seeds, 120 ep): coop rises with α (0.55→0.69
at α=0.2) and welfare falls (over-cleaning), matching the stored noisy regime.
The welfare-collapse motivates the stag mechanic — it gives cooperation something
*material* to reward.

## Key findings
- **Selfish learners fail one-shot.** In the threshold stag hunt (`s=5,h=3`,
  all-hands), independent IQL converges to risk-dominant Hare: coop `0.02±0.00`
  for N∈{2,3,4,5}. No temporal mechanism applies (horizon 1).
- **Cooperation self-organizes (n=2 bootstrap, 40 seeds).** Collaboration-based
  `A` bootstraps from `A=0`: learned caring `A_01` rises, cooperation follows. The
  one-shot stag hunt is **bimodal**, so we report **basin-crossing rate** with
  Wilson 95% CIs: collab **36/40** [0.77,0.96] ≈ fixed (`α=0.8`) **35/40**
  [0.74,0.95] ≫ selfish **0/40** [0.00,0.09]. Collab matches a *well-tuned* fixed
  coupling **with no tuned α**.
- **Partner-specific assortment / free-rider exclusion (40 seeds).** N=3, one
  frozen defector: collab learns `A_partner=0.92`, `A_defector=0.000` (exactly),
  while fixed wastes equal care `0.45/0.45`. Care is *symmetric by construction*
  (joint acts mutual → `C_ij=C_ji`). **Graded free-rider** (cooperates w.p. `p`):
  sharp exclusion at `p=0` (A→0) but `p≥0.25` already earns ~0.48 — the assortment
  is **near-binary**, not a continuous reciprocity meter (honest limitation).
- **Algorithm-independent (M3, 20 seeds, unified hyperparams).** Free-rider
  exclusion (A_def→0) is **universal** across IQL/DQN/A2C/PPO. Bootstrap is
  learner-dependent: IQL **17/20** ≈ fixed 16/20, PPO **20/20** = fixed, A2C
  partial (5/20), **DQN fails under ALL regimes** (collab≈fixed≈selfish≈3/20 — a
  learner limit, not a mechanism one).
- **Ablations.** (i) *Normalization is not the lever*: raw vs row-normalized
  `(I-A)^{-1}` give **identical** basin-crossing (21/24 each) tabular. (ii)
  *κ-robust*: ≥19/24 for κ≤1 (κ=0.5 default is conservative). (iii)
  *Orthogonality*: structural and temporal (repetition) levers **compose** —
  collab works at H=1 where repetition barely moves selfish (0.02→0.15), reaching
  ceiling (24/24) at H=10.
- **Embodied log-hunt (CleanupStag, A2C, 6 seeds).** Collab reaches `2.00±0.57`
  joint lifts/ep vs fixed `1.21±0.79` vs selfish `0.55±0.57`. Free-rider
  exclusion replicates: care among learners `≈0.42`, care to the frozen
  harvester `0.00` (fixed wastes `0.20`).
- **Feasibility ceiling.** All-hands cooperation is selectable for N≤3 but
  collapses for N≥4 under every regime as `α<1/(N−1)` caps the coupling while
  all-hands coordination becomes exponentially rare — an intrinsic limit with no
  temporal analogue.

## Reproduce (Part II)
```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy pytest
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
# matrix: bootstrap/freerider/groupsize/trajectory + revision sets
#   (graded free-rider, normalization ablation, kappa sweep, horizon composition)
PYTHONPATH=src .venv/bin/python -m prosocial.experiments.collab_matrix --set all --seeds 40
PYTHONPATH=src .venv/bin/python -m prosocial.experiments.collab_algos --seeds 20 --workers 12
PYTHONPATH=src .venv/bin/python -m prosocial.experiments.collab_spatial --seeds 6 --workers 8
PYTHONPATH=src .venv/bin/python -m prosocial.plotting_collab   # -> paper/figures/collab_*.png
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
# paper: cd paper && pdflatex acl_new && bibtex acl_new && pdflatex acl_new && pdflatex acl_new
```

> Note: `--set all` includes the `revision` sets; or run them alone with
> `--set revision`. The mechanism's canonical hyperparameters (`alpha_max=0.95,
> kappa=0.5, decay=0.98`) live in `CollaborationMatrix` and are used identically
> across all matrix experiments (the spatial env discloses its own values).

## For human review (where to look more) — written while you were away

The new paper (`paper/content_new.tex`) reached **Weak Accept** in two automated
ICML-style review rounds (`project/reviewer.md`, Opus). It is scientifically
sound for the stated objective, but it is *Weak* Accept, not Strong — places you
may want to push, in rough priority:

1. **Bimodality of the headline result.** The 2-player stag hunt is intrinsically
   bimodal (each seed → ~0.02 or ~0.98), so I report basin-crossing rates with
   Wilson CIs rather than means. If you want a *less* bimodal headline, consider a
   stag hunt with a smoother basin (e.g. continuous effort levels, or a softer
   risk-dominance gap `h` closer to `s/2`). Check `experiments/collab_matrix.py`
   `bootstrap_set` / `StagHuntN` params.
2. **Graded free-rider saturates (real limitation).** Care is near-binary: a
   25%-cooperator earns almost as much care as a 100%-cooperator
   (`results/collab_matrix_graded.jsonl`). To get *continuous* reciprocity-graded
   assortment, normalize the tally `C_ij` by co-action *opportunities* (a rate,
   not a count) or use a non-saturating link in `collaboration.py`
   `CollaborationMatrix.matrix()`. This is the single most impactful upgrade.
3. **DQN fails under all reward regimes** at our budget (a learner limit, not a
   mechanism one). If a reviewer wants 4/4 learners to *bootstrap*, DQN likely
   needs a longer budget / smaller replay / on-policy correction. See
   `experiments/collab_algos.py`.
4. **Spatial is smoke-scale (6 seeds, 400 ep).** The CleanupStag result is clean
   (collab 2.00 > fixed 1.21 > selfish 0.55 joint-lifts) but I deliberately did
   not scale it. For a benchmark-strength spatial claim, scale seeds/episodes and
   add CIs. Env: `envs/spatial.CleanupStag`.
5. **Exploitation claim is deferred, not shown.** The free-rider is *excluded*,
   but I do not demonstrate that a fixed-α "caring" agent is materially *exploited*
   (its payoff is constant in this stag hunt). A public-good variant where the
   free-rider benefits from others' cooperation would show real exploitation —
   worth adding if you want the AI-safety angle to be empirical, not predicted.
6. **Title/abstract framing.** I positioned this as orthogonal-to-Folk-Theorem and
   backed it with the horizon-composition experiment (§7). If you disagree with
   "orthogonal" vs "complementary/non-temporal", that is a wording call only you
   should make.

Nothing above blocks the current submission; these are the avenues to raise it
from Weak Accept toward Accept. All numbers in the paper are verified against the
`results/*.jsonl` (I re-audited and fixed one stale "12 seeds" → "20 seeds").
