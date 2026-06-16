"""Does the structure-vs-shaping gap generalize beyond one chain? (reviewer panels)

Multiple reviewers noted the structure!=shaping result rested on a single 3-chain.
Here we sweep FOUR graph topologies on a 4-player global Public Goods Game and
compare chain/graph interdependence U=(I-A)^{-1}pi against first-order-matched
neighbor shaping (beta=alpha) on each:
  - chain   (path 1-2-3-4)      : non-complete
  - ring    (4-cycle)           : non-complete
  - star    (hub + 3 leaves)    : non-complete
  - complete(K4)                : control -- shaping == structure to first order,
                                  expect ~no gap
For each topology we sweep alpha over its feasible range alpha < 1/rho(Adj) and
report the cooperation gap (interdep - neighbor-shaping). The prediction: a
positive gap on every NON-complete topology (indirect coupling helps), shrinking
toward zero on the complete graph. Writes results/topology.jsonl.
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


def _adj(topology):
    if topology == "chain":
        A = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], float)
    elif topology == "ring":
        A = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], float)
    elif topology == "star":
        A = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], float)
    elif topology == "complete":
        A = np.ones((4, 4)) - np.eye(4)
    else:
        raise ValueError(topology)
    return A


def _feasible_alphas(adj, k=8):
    rho = max(abs(np.linalg.eigvals(adj)))
    amax = 1.0 / rho
    return np.round(np.linspace(0.0, 0.95 * amax, k), 4), amax


def _run_one(job, episodes, mult):
    topology, kind, alpha, seed = job
    adj = _adj(topology)
    if alpha == 0:
        tf = Selfish()
    elif kind == "interdep":
        tf = GraphInterdependence(alpha * adj)
    else:
        tf = NeighborShaping(adj, beta=alpha)
    r = train_selfplay("PGG", tf, horizon=1, episodes=episodes, seed=seed,
                       game_kwargs={"n_agents": 4, "mult": mult})
    return dict(topology=topology, kind=kind, alpha=float(alpha), seed=seed,
                coop=r.coop_rate, welfare=r.social_welfare)


def run(seeds=12, episodes=3000, workers=8, mult=1.6, out="results/topology.jsonl"):
    jobs = []
    for topology in ["chain", "ring", "star", "complete"]:
        alphas, _ = _feasible_alphas(_adj(topology))
        for kind in ["interdep", "neighbor_shaping"]:
            for a in alphas:
                for s in range(seeds):
                    jobs.append((topology, kind, float(a), s))
    print(f"[topo] {len(jobs)} runs on {workers} workers (mult={mult})", flush=True)
    fn = partial(_run_one, episodes=episodes, mult=mult)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(outp, "w") as f, Pool(workers) as pool:
        for row in pool.imap_unordered(fn, jobs, chunksize=4):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
    # report max gap per topology
    for topology in ["chain", "ring", "star", "complete"]:
        byA = {}
        for r in rows:
            if r["topology"] == topology:
                byA.setdefault((r["kind"], r["alpha"]), []).append(r["coop"])
        alphas = sorted({a for (k, a) in byA})
        gaps = []
        for a in alphas:
            ci = byA.get(("interdep", a), [0]); ns = byA.get(("neighbor_shaping", a), [0])
            gaps.append(np.mean(ci) - np.mean(ns))
        amax = 1.0 / max(abs(np.linalg.eigvals(_adj(topology))))
        print(f"[topo] {topology:9s} (amax={amax:.3f}): max coop gap = {max(gaps):+.3f}", flush=True)
    print(f"[topo] wrote {len(rows)} rows -> {outp}")
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mult", type=float, default=1.6)
    args = ap.parse_args()
    run(seeds=args.seeds, episodes=args.episodes, workers=args.workers, mult=args.mult)
