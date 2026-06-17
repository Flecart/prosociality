"""Dissociating interdependence (alpha) from far-sightedness (gamma).

A 2x2 x xi factorial in the tunable-externality CommonsGame (envs/commons.py):

    mechanism cells:  alpha in {0, high} x gamma in {myopic 0, far-sighted}
    externality knob: xi in [0, 1]  (contemporaneous -> intertemporal)

Self-play A2C; each agent learns on the symmetric Bergstrom reward
U=(I-A)^{-1}pi with off-diagonal alpha, under discount gamma. The CommonsGame
observation is the stock only (no partner actions), so reciprocity is impossible
-- the only cooperation channels are the reward (alpha) and the stock value
(gamma). Cooperation = restraint (abstain fraction) on RAW behaviour.

Predicted double dissociation:
  * (alpha>0, gamma=0)  cooperates at xi=0  (interdependence cures the
    contemporaneous externality with no future) and fails at xi=1.
  * (alpha=0, gamma>0)  fails at xi=0 and cooperates at xi=1 (far-sightedness
    cures the intertemporal externality with no caring).
The two single-mechanism curves CROSS; that crossing rules out conflating the
two. Anchors: (0,0) defects everywhere; (high,high) cooperates everywhere.

Writes results/commons_dissociation.jsonl; figure in plotting_commons.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..agents.a2c import A2CAgent
from ..envs.commons import CommonsGame
from ..interdependence import symmetric_matrix, effective_utilities

ROOT = Path(__file__).resolve().parents[3]

CELLS = {
    "selfish_myopic":      dict(alpha=0.0, gamma=0.0),
    "interdep_myopic":     dict(alpha=0.8, gamma=0.0),
    "selfish_farsighted":  dict(alpha=0.0, gamma=0.95),
    "interdep_farsighted": dict(alpha=0.8, gamma=0.95),
}


def _train(xi, alpha, gamma, episodes, seed, device, n_agents=2,
           max_steps=60, lr=3e-3):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = CommonsGame(xi=xi, n_agents=n_agents, max_steps=max_steps, rng=rng)
    A = symmetric_matrix(n_agents, alpha)
    agents = [A2CAgent(env.obs_dim, env.n_actions, lr=lr, gamma=gamma,
                       device=device) for _ in range(n_agents)]
    coop = np.zeros(episodes)
    for ep in range(episodes):
        obs = env.reset()
        buf = [dict(lp=[], v=[], ent=[], r=[]) for _ in range(n_agents)]
        done = False
        while not done:
            acts, lps, vs, ents = [], [], [], []
            for i in range(n_agents):
                a, lp, v, e = agents[i].act(obs[i])
                acts.append(a); lps.append(lp); vs.append(v); ents.append(e)
            obs, pi, done = env.step(acts)
            u = effective_utilities(A, pi)            # learn on transformed reward
            for i in range(n_agents):
                buf[i]["lp"].append(lps[i]); buf[i]["v"].append(vs[i])
                buf[i]["ent"].append(ents[i]); buf[i]["r"].append(float(u[i]))
        for i in range(n_agents):
            agents[i].learn(buf[i]["lp"], buf[i]["v"], buf[i]["ent"], buf[i]["r"])
        coop[ep] = env.coop_signal()
    k = max(1, episodes // 10)
    return float(coop[-k:].mean())


def run(episodes=500, seeds=5, n_xi=5, out="results/commons_dissociation.jsonl",
        device=None, cells=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    xis = np.round(np.linspace(0.0, 1.0, n_xi), 3)
    sel = {k: v for k, v in CELLS.items() if cells is None or k in cells}
    print(f"[diss] device={device} cuda={torch.cuda.is_available()} "
          f"xis={list(xis)} episodes={episodes} seeds={seeds} cells={list(sel)}",
          flush=True)
    rows = []
    for name, cfg in sel.items():
        for xi in xis:
            cps = [_train(float(xi), cfg["alpha"], cfg["gamma"], episodes, s, device)
                   for s in range(seeds)]
            rows.append(dict(cell=name, alpha=cfg["alpha"], gamma=cfg["gamma"],
                             xi=float(xi), coop=float(np.mean(cps)),
                             coop_std=float(np.std(cps))))
            print(f"[diss] {name:20s} xi={xi:.2f}: coop={np.mean(cps):.2f} "
                  f"+/- {np.std(cps):.2f}", flush=True)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[diss] wrote {len(rows)} rows -> {outp}", flush=True)
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-xi", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cell", default=None, help="run a single cell (for parallelism)")
    ap.add_argument("--out", default="results/commons_dissociation.jsonl")
    args = ap.parse_args()
    run(episodes=args.episodes, seeds=args.seeds, n_xi=args.n_xi,
        out=args.out, device=args.device,
        cells=[args.cell] if args.cell else None)
