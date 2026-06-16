"""Structure vs. shaping on a NON-complete graph (addresses reviewer round 3).

Setup: a 3-player global Public Goods Game (each agent's contribution benefits
all three) whose agents are coupled on a CHAIN graph 1-2-3. We compare two
transforms matched at first order:
  - chain interdependence:  U = (I-A)^{-1} pi, A the chain adjacency * alpha
  - neighbor shaping:        r_i = pi_i + beta * sum_{j in neighbors(i)} pi_j, beta=alpha
The endpoints (agents 1 and 3) are NOT neighbors, so neighbor-shaping gives each
endpoint zero weight on the other; chain interdependence gives them an indirect
(alpha^2) stake through the center. If structure were merely first-order shaping,
the two would coincide. We test whether interdependence yields measurably higher
cooperation/welfare. Writes results/graph_structure.jsonl.

Feasibility: chain A has rho = alpha*sqrt(2), so alpha < 1/sqrt(2) ~ 0.707.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..rewards import GraphInterdependence, NeighborShaping, Selfish
from ..train import train_selfplay

ROOT = Path(__file__).resolve().parents[3]

# chain adjacency 1-2-3
ADJ = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)


def _run_one(job, episodes, mult):
    kind, alpha, seed = job
    if alpha == 0:
        tf = Selfish()
    elif kind == "chain_interdep":
        tf = GraphInterdependence(alpha * ADJ)
    else:  # neighbor_shaping, matched first order (beta = alpha)
        tf = NeighborShaping(ADJ, beta=alpha)
    r = train_selfplay("PGG", tf, horizon=1, episodes=episodes, seed=seed,
                       game_kwargs={"n_agents": 3, "mult": mult})
    return dict(kind=kind, alpha=float(alpha), seed=seed,
                coop=r.coop_rate, welfare=r.social_welfare)


def run(seeds=12, episodes=3000, workers=8, mult=1.6,
        out="results/graph_structure.jsonl"):
    alphas = np.round(np.linspace(0.0, 0.68, 9), 4)  # below 1/sqrt(2)
    jobs = [(k, float(a), s) for k in ("chain_interdep", "neighbor_shaping")
            for a in alphas for s in range(seeds)]
    print(f"[graph] {len(jobs)} runs on {workers} workers (mult={mult})", flush=True)
    fn = partial(_run_one, episodes=episodes, mult=mult)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(outp, "w") as f, Pool(workers) as pool:
        for row in pool.imap_unordered(fn, jobs, chunksize=4):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
    # report mean coop/welfare at the top alpha per kind
    top = max(a for a in {r["alpha"] for r in rows})
    for k in ("chain_interdep", "neighbor_shaping"):
        v = [r for r in rows if r["kind"] == k and abs(r["alpha"] - top) < 1e-6]
        mc = np.mean([r["coop"] for r in v]); mw = np.mean([r["welfare"] for r in v])
        print(f"[graph] {k} @ alpha={top}: coop={mc:.3f} welfare={mw:.3f}", flush=True)
    print(f"[graph] wrote {len(rows)} rows -> {outp}")
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mult", type=float, default=1.6)
    args = ap.parse_args()
    run(seeds=args.seeds, episodes=args.episodes, workers=args.workers, mult=args.mult)
