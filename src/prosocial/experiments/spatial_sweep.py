"""Smoke-scale spatial-dilemma sweep: coop signal + welfare vs interdependence.

Runs the A2C self-play loop on Coin Game / Harvest / Cleanup across a small alpha
grid (respecting the feasibility ceiling per N) and a few seeds, then writes
results/spatial.jsonl. Parallel over (env, alpha, seed). Deliberately modest --
this is plumbing-scale evidence, not a tuned benchmark.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

# many Pool workers x torch intra-op threads oversubscribe and hang the node
torch.set_num_threads(1)

from ..rewards import Interdependence, Selfish
from ..train_spatial import train_spatial

ROOT = Path(__file__).resolve().parents[3]

ENV_N = {"CoinGame": 2, "Harvest": 4, "Cleanup": 4}


def _alphas(n, k):
    amax = 1.0 / (n - 1)
    return np.round(np.linspace(0.0, 0.9 * amax, k), 3)


def _run_one(job, episodes):
    env, n, alpha, seed = job
    tf = Selfish() if alpha == 0 else Interdependence(n, alpha)
    r = train_spatial(env, tf, episodes=episodes, seed=seed)
    return dict(env=env, n_agents=n, alpha=float(alpha), seed=seed,
                coop=r.coop_signal, welfare=r.social_welfare)


def run(seeds=3, episodes=120, k_alpha=4, workers=6,
        out="results/spatial.jsonl"):
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    # resume: skip (env, alpha, seed) already present (survives preemption)
    done = set()
    if outp.exists():
        for line in open(outp):
            try:
                r = json.loads(line)
                done.add((r["env"], round(r["alpha"], 4), r["seed"]))
            except Exception:
                pass
    jobs = []
    for env, n in ENV_N.items():
        for a in _alphas(n, k_alpha):
            for s in range(seeds):
                if (env, round(float(a), 4), s) not in done:
                    jobs.append((env, n, float(a), s))
    print(f"[spatial] {len(jobs)} runs to do ({len(done)} already done) "
          f"on {workers} workers (episodes={episodes})", flush=True)
    fn = partial(_run_one, episodes=episodes)
    n_done = len(done)
    with open(outp, "a") as f, Pool(workers) as pool:   # append -> resumable
        for i, row in enumerate(pool.imap_unordered(fn, jobs), 1):
            f.write(json.dumps(row) + "\n"); f.flush()
            n_done += 1
            print(f"[spatial] {i}/{len(jobs)} ({n_done} total): {row['env']} "
                  f"a={row['alpha']} coop={row['coop']:.2f} welf={row['welfare']:.1f}",
                  flush=True)
    print(f"[spatial] wrote -> {outp} ({n_done} total rows)")
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--k-alpha", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run(seeds=args.seeds, episodes=args.episodes, k_alpha=args.k_alpha,
        workers=args.workers)
