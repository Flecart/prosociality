"""Collaboration-based endogenous interdependence on the matrix StagHuntN (M2).

Independent tabular Q-learners play the one-shot N-player threshold stag hunt
(no repetition, so the folk theorem cannot operate). We compare four reward
regimes on the SAME learners:

  * selfish        -- A = 0 (the risk-dominant trap; learners settle on Hare).
  * fixed-interdep -- A = symmetric alpha (exogenous, uniform care).
  * collab         -- A built online from observed joint Stag hunts
                      (CollaborationMatrix): the M2 mechanism.
  * (free-rider)   -- N=3 with one frozen defector; does collab concentrate
                      care on the reciprocating partner and starve the defector?

Cooperation/welfare are measured on RAW payoffs. We log the learned A so we can
inspect symmetry and partner-specificity. Writes results/collab_matrix_*.jsonl.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ..agents import TabularQLearner
from ..collaboration import CollaborationMatrix, safe_A
from ..envs import RepeatedGame, make_game
from ..interdependence import (
    effective_utilities,
    normalized_effective_utilities,
    symmetric_matrix,
)

ROOT = Path(__file__).resolve().parents[3]


def _transform(A, pi, normalize=True):
    if normalize:
        return normalized_effective_utilities(A, pi)
    return effective_utilities(A, pi)


def _costag_pairs(actions, min_staggers):
    """Pairs (i,j) that jointly hunted Stag in a *successful* hunt this step."""
    staggers = [i for i, a in enumerate(actions) if a == 0]
    if len(staggers) < min_staggers:
        return []
    return list(itertools.combinations(staggers, 2))


def run(mode="collab", n=2, stag=5.0, hare=3.0, min_staggers=None,
        episodes=3000, seed=0, alpha_fixed=0.6, alpha_max=0.95, kappa=0.5,
        decay=0.98, fixed_defectors=None, normalize=True, record_every=20,
        horizon=1, defector_coop_p=0.0):
    """One run; returns dict with coop/welfare curves and A trajectory.

    Cooperation is reported over the *learners* (frozen defectors excluded) so a
    free-rider does not deflate the metric. The collab link defaults
    (alpha_max=0.95, kappa=0.5, decay=0.98) are tuned so the positive feedback
    reliably clears the stag-hunt selection threshold once seeded.
    """
    min_staggers = n if min_staggers is None else min_staggers
    rng = np.random.default_rng(seed)
    gk = dict(n_agents=n, stag=stag, hare=hare, min_staggers=min_staggers)
    base = make_game("StagHuntN", **gk)
    env = RepeatedGame(base, horizon=horizon)
    fixed_defectors = set(fixed_defectors or [])
    learners = [i for i in range(n) if i not in fixed_defectors]

    agents = [TabularQLearner(env.n_states, env.n_actions, rng=rng) for _ in range(n)]
    collab = CollaborationMatrix(n, alpha_max=alpha_max, kappa=kappa, decay=decay)
    # the fixed baseline must respect the spectral ceiling rho(A)<1 too
    A_fixed = safe_A(symmetric_matrix(n, alpha_fixed)) if mode == "fixed" else None

    coop_curve = np.zeros(episodes)        # learner cooperation rate
    welfare_curve = np.zeros(episodes)     # total raw payoff
    payoff_accum = np.zeros(n)
    A_traj = []

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        for ag in agents:
            ag.set_epsilon(frac)
        if mode == "selfish":
            A = np.zeros((n, n))
        elif mode == "fixed":
            A = A_fixed
        else:  # collab
            A = collab.matrix()

        state = env.reset()
        done = False
        ep_coop = 0
        ep_welfare = 0.0
        while not done:
            actions = []
            for i in range(n):
                if i in fixed_defectors:
                    # frozen free-rider: Stag (cooperate) with prob defector_coop_p,
                    # else Hare. defector_coop_p=0 is the pure always-defect case;
                    # a graded value tests whether collab adaptively down-weights a
                    # partner that only *sometimes* reciprocates.
                    actions.append(0 if rng.random() < defector_coop_p else 1)
                else:
                    actions.append(agents[i].act(state))
            next_state, pi, done = env.step(actions)
            r = _transform(A, pi, normalize) if A.any() else pi
            for i in learners:
                agents[i].update(state, actions[i], r[i], next_state, done)
            collab.observe_pairs(_costag_pairs(actions, min_staggers))
            state = next_state
            ep_coop += sum(int(actions[i] == 0) for i in learners)
            ep_welfare += pi.sum()
            payoff_accum += pi
        collab.end_episode()
        coop_curve[ep] = ep_coop / max(1, len(learners))
        welfare_curve[ep] = ep_welfare
        if ep % record_every == 0 or ep == episodes - 1:
            A_traj.append([ep] + [round(float(A[i, j]), 4)
                                  for i in range(n) for j in range(n)])

    k = max(1, episodes // 10)
    A_final = (collab.matrix() if mode == "collab"
               else (A_fixed if mode == "fixed" else np.zeros((n, n))))
    return dict(
        mode=mode, n=n, seed=seed, stag=stag, hare=hare,
        min_staggers=min_staggers, alpha_fixed=alpha_fixed,
        coop_rate=float(coop_curve[-k:].mean()),
        coop_std=float(coop_curve[-k:].std()),
        social_welfare=float(welfare_curve[-k:].mean()),
        learner_payoff=float(payoff_accum[learners].sum() / len(learners) / episodes),
        raw_payoff=[round(float(payoff_accum[i] / episodes), 3) for i in range(n)],
        A_final=[[round(float(A_final[i, j]), 4) for j in range(n)] for i in range(n)],
        coop_curve=[round(float(x), 4) for x in coop_curve],
        A_traj=A_traj,
    )


# ---- experiment sets -------------------------------------------------------

def bootstrap_set(seeds=12, episodes=3000, out="results/collab_matrix_bootstrap.jsonl"):
    """Core: 2-player log-hunt (min_staggers=2). selfish vs fixed vs collab.

    Selfish reliably fails (risk-dominant Hare); fixed-interdep is a coordination
    gamble; collab self-organizes the coupling from observed joint hunts.
    """
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("selfish", "fixed", "collab"):
        for s in range(seeds):
            r = run(mode=mode, n=2, min_staggers=2, hare=3.0, episodes=episodes,
                    seed=s, alpha_fixed=0.8)
            r.pop("coop_curve"); r.pop("A_traj")   # keep file small
            rows.append(r)
            print(f"[boot] {mode:8s} s={s} coop={r['coop_rate']:.2f} "
                  f"welf={r['social_welfare']:.2f} A01={r['A_final'][0][1]:.2f}",
                  flush=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[boot] wrote {len(rows)} rows -> {outp}")


def groupsize_set(seeds=12, episodes=3000, out="results/collab_matrix_groupsize.jsonl"):
    """'All hands' stag hunt (min_staggers=N): selfish fails across N; does collab
    scale? Uses fixed alpha at 0.9*alpha_max(N) for a feasibility-respecting baseline."""
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (2, 3, 4, 5):
        a_fixed = round(0.9 / (n - 1), 3)   # below the spectral ceiling 1/(N-1)
        for mode in ("selfish", "fixed", "collab"):
            for s in range(seeds):
                r = run(mode=mode, n=n, min_staggers=n, hare=3.0, episodes=episodes,
                        seed=s, alpha_fixed=a_fixed)
                r.pop("coop_curve"); r.pop("A_traj")
                rows.append(r)
                print(f"[grp] n={n} {mode:8s} s={s} coop={r['coop_rate']:.2f}",
                      flush=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[grp] wrote {len(rows)} rows -> {outp}")


def freerider_set(seeds=10, episodes=3000, out="results/collab_matrix_freerider.jsonl"):
    """N=3, agent 2 a frozen defector; min_staggers=2 so the 2 learners can hunt.

    Does collab concentrate care on the reciprocating partner (A_01) and starve
    the defector (A_02 -> 0), beating fixed-interdep which wastes care on it?
    """
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("selfish", "fixed", "collab"):
        for s in range(seeds):
            r = run(mode=mode, n=3, min_staggers=2, hare=3.0, episodes=episodes,
                    seed=s, fixed_defectors=[2], alpha_fixed=0.45)
            r.pop("coop_curve"); r.pop("A_traj")
            rows.append(r)
            A = np.array(r["A_final"])
            print(f"[free] {mode:8s} s={s} coop={r['coop_rate']:.2f} "
                  f"A01={A[0,1]:.2f} A02={A[0,2]:.2f} pay={r['learner_payoff']:.2f}",
                  flush=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[free] wrote {len(rows)} rows -> {outp}")


def trajectory_set(seeds=10, episodes=3000, out="results/collab_matrix_traj.jsonl"):
    """Full curves + A trajectory for the n=2 bootstrap (for the time-series plot)."""
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("selfish", "collab"):
        for s in range(seeds):
            r = run(mode=mode, n=2, min_staggers=2, episodes=episodes, seed=s)
            rows.append(r)
            print(f"[traj] n=2 {mode:8s} s={s} coop={r['coop_rate']:.2f}", flush=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[traj] wrote {len(rows)} rows -> {outp}")


def graded_freerider_set(seeds=12, episodes=3000,
                         out="results/collab_matrix_graded.jsonl"):
    """Graded free-rider: a partner that cooperates with probability p in {0, .25,
    .5, .75, 1}. Tests whether collab's care toward it *tracks* its reciprocation
    (emergent), rather than the always-defect p=0 case where A=0 is definitional."""
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        for s in range(seeds):
            r = run(mode="collab", n=3, min_staggers=2, hare=3.0, episodes=episodes,
                    seed=s, fixed_defectors=[2], defector_coop_p=p)
            r.pop("coop_curve"); r.pop("A_traj")
            r["defector_coop_p"] = p
            rows.append(r)
            A = np.array(r["A_final"])
            print(f"[graded] p={p} s={s} A_partner={A[0,1]:.2f} "
                  f"A_freerider={A[0,2]:.2f}", flush=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[graded] wrote {len(rows)} rows -> {outp}")


def ablation_set(seeds=24, episodes=3000, out="results/collab_matrix_ablation.jsonl"):
    """Normalization ablation: bootstrap under the NORMALIZED vs RAW (I-A)^{-1}
    transform, to show the cooperation lift is not an artifact of row-normalization."""
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for normalize in (True, False):
        for mode in ("selfish", "fixed", "collab"):
            for s in range(seeds):
                r = run(mode=mode, n=2, min_staggers=2, hare=3.0, episodes=episodes,
                        seed=s, alpha_fixed=0.8, normalize=normalize)
                r.pop("coop_curve"); r.pop("A_traj")
                r["normalize"] = normalize
                rows.append(r)
        print(f"[ablation] normalize={normalize} done", flush=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[ablation] wrote {len(rows)} rows -> {outp}")


def kappa_sweep_set(seeds=24, episodes=3000, out="results/collab_matrix_kappa.jsonl"):
    """Sensitivity of the bootstrap to the link half-saturation kappa."""
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for kappa in (0.25, 0.5, 1.0, 2.0, 4.0):
        for s in range(seeds):
            r = run(mode="collab", n=2, min_staggers=2, hare=3.0, episodes=episodes,
                    seed=s, kappa=kappa)
            r.pop("coop_curve"); r.pop("A_traj")
            r["kappa"] = kappa
            rows.append(r)
        frac = np.mean([row["coop_rate"] > 0.5 for row in rows if row["kappa"] == kappa])
        print(f"[kappa] kappa={kappa} basin-crossing frac={frac:.2f}", flush=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[kappa] wrote {len(rows)} rows -> {outp}")


def composition_set(seeds=24, episodes=3000, out="results/collab_matrix_horizon.jsonl"):
    """Orthogonality to the Folk Theorem: 2x2 of {selfish, collab} x horizon
    {1, 10}. Repetition (a temporal lever) and collaboration (a structural lever)
    should each add cooperation and compose, rather than substitute."""
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for horizon in (1, 10):
        for mode in ("selfish", "collab"):
            for s in range(seeds):
                r = run(mode=mode, n=2, min_staggers=2, hare=3.0, episodes=episodes,
                        seed=s, horizon=horizon)
                r.pop("coop_curve"); r.pop("A_traj")
                r["horizon"] = horizon
                rows.append(r)
            cr = np.mean([row["coop_rate"] for row in rows
                          if row["horizon"] == horizon and row["mode"] == mode])
            print(f"[horizon] H={horizon} {mode}: coop={cr:.2f}", flush=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[horizon] wrote {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["bootstrap", "groupsize", "freerider",
                                      "trajectory", "graded", "ablation", "kappa",
                                      "composition", "all", "revision"],
                    default="all")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=3000)
    args = ap.parse_args()
    if args.set in ("bootstrap", "all"):
        bootstrap_set(seeds=args.seeds, episodes=args.episodes)
    if args.set in ("groupsize", "all"):
        groupsize_set(seeds=args.seeds, episodes=args.episodes)
    if args.set in ("freerider", "all"):
        freerider_set(seeds=args.seeds, episodes=args.episodes)
    if args.set in ("trajectory", "all"):
        trajectory_set(seeds=args.seeds, episodes=args.episodes)
    if args.set in ("graded", "all", "revision"):
        graded_freerider_set(seeds=args.seeds, episodes=args.episodes)
    if args.set in ("ablation", "all", "revision"):
        ablation_set(seeds=args.seeds, episodes=args.episodes)
    if args.set in ("kappa", "all", "revision"):
        kappa_sweep_set(seeds=args.seeds, episodes=args.episodes)
    if args.set in ("composition", "all", "revision"):
        composition_set(seeds=args.seeds, episodes=args.episodes)
