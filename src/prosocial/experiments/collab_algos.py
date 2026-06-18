"""Algorithm comparison for collaboration-based interdependence (M3).

Does cooperation-from-interdependence emerge regardless of the underlying MARL
algorithm? We run the SAME collaboration mechanism (CollaborationMatrix building
A from observed joint Stag hunts) on the one-shot matrix StagHuntN under four
independent learners:

  * IQL  -- tabular Q-learning   (off-policy, value, exact table)
  * DQN  -- deep Q-network       (off-policy, value, neural + replay + target)
  * A2C  -- advantage actor-critic (on-policy, policy-gradient, neural)
  * PPO  -- clipped PPO          (on-policy, policy-gradient, neural)

For each learner we report selfish / fixed / collab on the n=2 bootstrap and the
n=3 free-rider (care concentration). The loop is one-shot (horizon=1), so every
episode is a single decision: off-policy learners update per step, on-policy
learners (A2C/PPO) learn on the length-1 trajectory / buffered batch.
Writes results/collab_algos.jsonl.
"""

from __future__ import annotations

import argparse
import itertools
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from ..collaboration import CollaborationMatrix, safe_A
from ..envs import make_game
from ..interdependence import normalized_effective_utilities, symmetric_matrix

ROOT = Path(__file__).resolve().parents[3]

LEARNERS = ("iql", "dqn", "a2c", "ppo")


def _make_agent(kind, n_states, n_actions, rng, seed):
    if kind == "iql":
        from ..agents import TabularQLearner
        return TabularQLearner(n_states, n_actions, rng=rng)
    if kind == "dqn":
        from ..agents import DQNLearner
        return DQNLearner(n_states, n_actions, rng=rng, device="cpu")
    if kind == "ppo":
        from ..agents import PPOLearner
        return PPOLearner(n_states, n_actions, rng=rng, device="cpu", batch=128)
    if kind == "a2c":
        from ..agents import A2CAgent
        return A2CAgent(n_states, n_actions)   # uses one-hot obs we pass in
    raise ValueError(kind)


def _costag_pairs(actions, min_staggers):
    staggers = [i for i, a in enumerate(actions) if a == 0]
    if len(staggers) < min_staggers:
        return []
    return list(itertools.combinations(staggers, 2))


def run(kind, mode="collab", n=2, min_staggers=2, hare=3.0, stag=5.0,
        episodes=3000, seed=0, alpha_fixed=0.8, fixed_defectors=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    base = make_game("StagHuntN", n_agents=n, stag=stag, hare=hare,
                     min_staggers=min_staggers)
    n_states = 1                       # trivial one-shot state
    fixed_defectors = set(fixed_defectors or [])
    learners = [i for i in range(n) if i not in fixed_defectors]
    agents = [_make_agent(kind, n_states, base.n_actions, rng, seed + i)
              for i in range(n)]
    collab = CollaborationMatrix(n)
    A_fixed = safe_A(symmetric_matrix(n, alpha_fixed)) if mode == "fixed" else None
    onehot = np.array([1.0], dtype=np.float32)   # obs for neural learners

    coop_curve = np.zeros(episodes)
    payoff_accum = np.zeros(n)

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        if mode == "selfish":
            A = np.zeros((n, n))
        elif mode == "fixed":
            A = A_fixed
        else:
            A = collab.matrix()

        actions = [None] * n
        ac_cache = {}
        for i in range(n):
            if i in fixed_defectors:
                actions[i] = 1
            elif kind == "a2c":
                a, lp, v, ent = agents[i].act(onehot)
                actions[i] = a
                ac_cache[i] = (lp, v, ent)
            else:
                agents[i].set_epsilon(frac)
                actions[i] = agents[i].act(0)
        pi = base.payoffs(actions)
        r = normalized_effective_utilities(A, pi) if A.any() else pi
        for i in learners:
            if kind == "a2c":
                lp, v, ent = ac_cache[i]
                agents[i].learn([lp], [v], [ent], [float(r[i])])
            else:
                agents[i].update(0, actions[i], float(r[i]), 0, True)
        collab.observe_pairs(_costag_pairs(actions, min_staggers))
        collab.end_episode()
        coop_curve[ep] = sum(int(actions[i] == 0) for i in learners) / len(learners)
        payoff_accum += pi

    if kind == "ppo":
        for ag in agents:
            if hasattr(ag, "flush"):
                ag.flush()
    k = max(1, episodes // 10)
    A_final = (collab.matrix() if mode == "collab"
               else (A_fixed if mode == "fixed" else np.zeros((n, n))))
    return dict(learner=kind, mode=mode, n=n, seed=seed,
                coop_rate=float(coop_curve[-k:].mean()),
                A_final=[[round(float(A_final[i, j]), 4) for j in range(n)]
                         for i in range(n)])


def _job(args):
    kind, mode, setting, seed = args
    if setting == "bootstrap":
        r = run(kind, mode=mode, n=2, min_staggers=2, hare=3.0, episodes=3000,
                seed=seed, alpha_fixed=0.8)
    else:  # freerider
        r = run(kind, mode=mode, n=3, min_staggers=2, hare=3.0, episodes=3000,
                seed=seed, alpha_fixed=0.45, fixed_defectors=[2])
    r["setting"] = setting
    return r


def main(seeds=8, workers=10, out="results/collab_algos.jsonl"):
    jobs = [(k, m, setting, s)
            for k in LEARNERS
            for setting in ("bootstrap", "freerider")
            for m in ("selfish", "fixed", "collab")
            for s in range(seeds)]
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    print(f"[algos] {len(jobs)} runs on {workers} workers", flush=True)
    rows = []
    with open(outp, "w") as f, Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 20 == 0:
                print(f"[algos] {i}/{len(jobs)}", flush=True)
    print(f"[algos] wrote {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    main(seeds=args.seeds, workers=args.workers)
