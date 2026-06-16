"""On-policy (A2C) confirmation of the one-shot IPD transition (reviewer round 4).

The reviewer asks whether the one-shot transition survives a non-tabular,
on-policy learner rather than epsilon-greedy IQL. We re-run the one-shot IPD
alpha-sweep with independent neural advantage actor-critic (A2C) agents -- the
same learner used for the spatial dilemmas -- on a trivial one-step observation,
and check the transition persists. Writes results/a2c_ipd.jsonl.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

# Tiny MLPs across many Pool workers: pin to 1 thread/process to avoid
# torch intra-op threads x workers oversubscribing the node (which hangs it).
torch.set_num_threads(1)

from ..agents.a2c import A2CAgent
from ..envs import make_game
from ..rewards import Interdependence, Selfish

ROOT = Path(__file__).resolve().parents[3]


def _train_ipd_a2c(alpha, seed, episodes=4000, lr=5e-3):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    game = make_game("IPD")
    tf = Selfish() if alpha == 0 else Interdependence(2, alpha)
    obs = np.array([1.0], dtype=np.float32)  # trivial one-shot observation
    agents = [A2CAgent(1, 2, lr=lr) for _ in range(2)]
    coop_idx = game.coop_action
    coop_curve = np.zeros(episodes)
    for ep in range(episodes):
        acts, lps, vs, ents = [], [], [], []
        for i in range(2):
            a, lp, v, ent = agents[i].act(obs)
            acts.append(a); lps.append(lp); vs.append(v); ents.append(ent)
        pi = game.payoffs(acts)
        u = tf(pi)
        for i in range(2):
            agents[i].learn([lps[i]], [vs[i]], [ents[i]], [float(u[i])])
        coop_curve[ep] = sum(int(a == coop_idx) for a in acts) / 2
    k = max(1, episodes // 10)
    return float(coop_curve[-k:].mean())


def _run_one(job, episodes):
    alpha, seed = job
    return dict(learner="a2c", alpha=float(alpha), seed=seed,
                coop=_train_ipd_a2c(alpha, seed, episodes=episodes))


def run(seeds=10, episodes=4000, workers=8, out="results/a2c_ipd.jsonl"):
    alphas = np.round(np.linspace(0.0, 0.9, 10), 3)
    jobs = [(float(a), s) for a in alphas for s in range(seeds)]
    print(f"[a2c] {len(jobs)} runs on {workers} workers", flush=True)
    fn = partial(_run_one, episodes=episodes)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(outp, "w") as f, Pool(workers) as pool:
        for row in pool.imap_unordered(fn, jobs, chunksize=2):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
    byA = {}
    for r in rows:
        byA.setdefault(r["alpha"], []).append(r["coop"])
    onset = next((a for a in sorted(byA) if np.mean(byA[a]) > 0.5), None)
    print(f"[a2c] IPD one-shot onset(coop>0.5)={onset}", flush=True)
    for a in sorted(byA):
        print(f"[a2c]   alpha={a}: coop={np.mean(byA[a]):.2f}", flush=True)
    print(f"[a2c] wrote {len(rows)} rows -> {outp}")
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(seeds=args.seeds, episodes=args.episodes, workers=args.workers)
