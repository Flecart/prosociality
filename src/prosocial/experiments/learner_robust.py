"""Learner-robustness of the one-shot transition (addresses reviewer round 2).

The reviewer notes the IPD H=1 transition location could be an artifact of
epsilon-greedy IQL exploration. We re-run the one-shot IPD alpha sweep under
three different learners/exploration schemes and check that a sharp transition
persists (its *location* may shift, but its *existence* should not depend on the
specific learner). Writes results/learner_robust.jsonl.

Learners:
  - egreedy_anneal : default, eps 0.5 -> 0.02
  - egreedy_low    : weak exploration, eps 0.1 -> 0.0
  - boltzmann      : softmax action selection, temperature 0.3
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..rewards import Interdependence, Selfish
from ..train import train_selfplay

ROOT = Path(__file__).resolve().parents[3]

LEARNERS = {
    "egreedy_anneal": dict(policy="egreedy", eps_start=0.5, eps_end=0.02),
    "egreedy_low": dict(policy="egreedy", eps_start=0.1, eps_end=0.0),
    "boltzmann": dict(policy="boltzmann", temperature=0.3),
}


def _run_one(job, episodes):
    learner, alpha, seed = job
    tf = Selfish() if alpha == 0 else Interdependence(2, alpha)
    r = train_selfplay("IPD", tf, horizon=1, episodes=episodes, seed=seed,
                       agent_kwargs=LEARNERS[learner])
    return dict(learner=learner, alpha=float(alpha), seed=seed, coop=r.coop_rate)


def run(seeds=10, episodes=3000, workers=8, out="results/learner_robust.jsonl"):
    alphas = np.round(np.linspace(0.0, 0.9, 10), 3)
    jobs = [(L, float(a), s) for L in LEARNERS for a in alphas for s in range(seeds)]
    print(f"[robust] {len(jobs)} runs on {workers} workers", flush=True)
    fn = partial(_run_one, episodes=episodes)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(outp, "w") as f, Pool(workers) as pool:
        for row in pool.imap_unordered(fn, jobs, chunksize=4):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
    # report onset per learner
    for L in LEARNERS:
        byA = {}
        for r in rows:
            if r["learner"] == L:
                byA.setdefault(r["alpha"], []).append(r["coop"])
        onset = next((a for a in sorted(byA) if np.mean(byA[a]) > 0.5), None)
        print(f"[robust] {L}: onset(coop>0.5)={onset}", flush=True)
    print(f"[robust] wrote {len(rows)} rows -> {outp}")
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(seeds=args.seeds, episodes=args.episodes, workers=args.workers)
