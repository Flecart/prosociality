"""Group-size feasibility law across N (addresses reviewer: test at >1 group size).

For the N-player Public Goods Game we sweep the symmetric coupling alpha over its
feasible range [0, 1/(N-1)) for several N, one-shot, and locate the empirical
cooperation onset (first alpha with mean coop > 0.5). The prediction is that the
onset tracks the spectral ceiling alpha_max(N) = 1/(N-1): larger groups both have
a lower ceiling and must sit closer to it before cooperation is structurally
optimal. Writes results/group_size.jsonl.
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

NS = [2, 3, 4, 6, 8]


def _alphas(n, k=10):
    amax = 1.0 / (n - 1)
    return np.round(np.linspace(0.0, 0.97 * amax, k), 4)


def _run_one(job, episodes):
    n, alpha, seed = job
    tf = Selfish() if alpha == 0 else Interdependence(n, alpha)
    r = train_selfplay("PGG", tf, horizon=1, episodes=episodes, seed=seed,
                       game_kwargs={"n_agents": n})
    return dict(n_agents=n, alpha=float(alpha), alpha_max=1.0 / (n - 1),
                seed=seed, coop=r.coop_rate, welfare=r.social_welfare)


def run(seeds=8, episodes=4000, workers=4, out="results/group_size.jsonl"):
    jobs = [(n, float(a), s) for n in NS for a in _alphas(n) for s in range(seeds)]
    print(f"[group] {len(jobs)} runs on {workers} workers", flush=True)
    fn = partial(_run_one, episodes=episodes)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(outp, "w") as f, Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(fn, jobs, chunksize=4), 1):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
    # report empirical onset per N
    for n in NS:
        sub = [r for r in rows if r["n_agents"] == n]
        byA = {}
        for r in sub:
            byA.setdefault(r["alpha"], []).append(r["coop"])
        onset = next((a for a in sorted(byA) if np.mean(byA[a]) > 0.5), None)
        print(f"[group] N={n}: alpha_max={1/(n-1):.3f}  empirical onset(coop>0.5)={onset}",
              flush=True)
    print(f"[group] wrote {len(rows)} rows -> {outp}")
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run(seeds=args.seeds, episodes=args.episodes, workers=args.workers)
