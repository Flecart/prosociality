"""Experiment 1: cooperation phase transition under fixed interdependence.

Sweeps the symmetric coupling alpha (and the reward-shaping beta baseline)
across games, horizons and seeds, recording cooperation / welfare / gini on RAW
payoffs. Writes results/phase_transition.jsonl.

Respects the spectral-radius constraint: for symmetric A, rho(A) = (N-1)*alpha,
so alpha is only feasible below 1/(N-1). We sweep each game over its feasible
range (a finding in itself: feasible interdependence shrinks with group size).
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..rewards import Interdependence, RewardShaping, Selfish
from ..train import train_selfplay

ROOT = Path(__file__).resolve().parents[3]

# (game, n_agents, horizons). alpha grid is derived from the spectral constraint.
GAME_SPECS = {
    "IPD": dict(n=2, horizons=[1, 5, 10, 100]),
    "StagHunt": dict(n=2, horizons=[1, 100]),
    "PGG": dict(n=4, horizons=[1, 100]),
}


def feasible_alphas(n_agents, n=10):
    """Grid of alpha in [0, alpha_max) with alpha_max = 1/(n_agents-1)."""
    amax = 1.0 / (n_agents - 1)
    # keep a margin below the singularity
    hi = min(0.9, 0.95 * amax)
    return np.round(np.linspace(0.0, hi, n), 4)


def _build_jobs(seeds, quick):
    """Enumerate (game, n, H, family, param, seed) job specs."""
    jobs = []
    for game, spec in GAME_SPECS.items():
        n = spec["n"]
        alphas = feasible_alphas(n, n=6 if quick else 10)
        betas = np.round(np.linspace(0.0, 1.0, 6 if quick else 8), 4)
        horizons = spec["horizons"][:2] if quick else spec["horizons"]
        for H in horizons:
            for a in alphas:
                for seed in range(seeds):
                    jobs.append((game, n, H, "interdep", float(a), seed))
            if H in (1, 100):
                for b in betas:
                    for seed in range(seeds):
                        jobs.append((game, n, H, "shaping", float(b), seed))
    return jobs


def _run_one(job, episodes):
    game, n, H, family, param, seed = job
    if family == "interdep":
        tf = Selfish() if param == 0 else Interdependence(n, param)
    else:
        tf = Selfish() if param == 0 else RewardShaping(param)
    r = train_selfplay(game, tf, horizon=H, episodes=episodes, seed=seed)
    return dict(game=game, n_agents=n, horizon=H, family=family, param=param,
                seed=seed, coop=r.coop_rate, welfare=r.social_welfare,
                gini=r.gini, stability=r.coop_stability)


def run(seeds=10, episodes=5000, out="results/phase_transition.jsonl",
        quick=False, workers=8):
    if quick:
        seeds, episodes = 3, 1500
    jobs = _build_jobs(seeds, quick)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    print(f"[phase] {len(jobs)} runs on {workers} workers (episodes={episodes})", flush=True)
    fn = partial(_run_one, episodes=episodes)
    n = 0
    # stream rows to disk as they complete -> crash-resilient, no end-of-run loss
    with open(outp, "w") as f, Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(fn, jobs, chunksize=4), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            n = i
            if i % 50 == 0 or i == len(jobs):
                print(f"[phase] {i}/{len(jobs)} done", flush=True)
    print(f"[phase] wrote {n} rows -> {outp}")
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/phase_transition.jsonl")
    args = ap.parse_args()
    run(seeds=args.seeds, episodes=args.episodes, out=args.out,
        quick=args.quick, workers=args.workers)
