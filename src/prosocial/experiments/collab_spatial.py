"""Collaboration-based interdependence in the spatial log-hunt (M2, embodied).

The same mechanism as collab_matrix, now in the temporally + spatially extended
CleanupStag gridworld: A is rebuilt each episode from the per-pair joint log-lift
tally the env records (env.step_colift_pairs), fed through GraphInterdependence
(normalized) into the A2C self-play loop. The reward transform is the only
coupling point, and -- unlike the platform's fixed-transform train_spatial -- it
is recomputed every episode from observed cooperation.

Conditions: selfish (A=0), fixed symmetric Interdependence(alpha), and collab
(A from joint lifts). We also run a free-rider variant where one agent is a
frozen apple-harvester (never lifts): does collab withhold care from it while
fixed-interdep wastes care on the exploiter? Metrics (joint-lift rate, welfare,
coop share) are on RAW payoffs. Writes results/collab_spatial*.jsonl.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from ..agents.a2c import A2CAgent
from ..collaboration import CollaborationMatrix
from ..envs.spatial import CleanupStag
from ..interdependence import normalized_effective_utilities, symmetric_matrix
from ..collaboration import safe_A

ROOT = Path(__file__).resolve().parents[3]


def _transform(A, pi):
    return normalized_effective_utilities(A, pi) if A.any() else np.asarray(pi, float)


def train(mode="collab", n=4, episodes=400, seed=0, alpha_fixed=0.2,
          lr=3e-3, env_kwargs=None, frozen_harvester=None,
          alpha_max=0.9, kappa=1.0, decay=0.95):
    """A2C self-play on CleanupStag with an episode-updated collaboration A.

    frozen_harvester: agent index that always harvests/never lifts (a free-rider
    on the cooperative stag). It still occupies the grid and eats apples.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = CleanupStag(n_agents=n, rng=rng, **(env_kwargs or {}))
    agents = [A2CAgent(env.obs_dim, env.n_actions, lr=lr) for _ in range(n)]
    collab = CollaborationMatrix(n, alpha_max=alpha_max, kappa=kappa, decay=decay)
    A_fixed = safe_A(symmetric_matrix(n, alpha_fixed)) if mode == "fixed" else None
    frozen = set([frozen_harvester] if frozen_harvester is not None else [])
    learners = [i for i in range(n) if i not in frozen]

    jl_curve = np.zeros(episodes)
    wf_curve = np.zeros(episodes)
    coop_curve = np.zeros(episodes)
    A_traj = []

    for ep in range(episodes):
        if mode == "selfish":
            A = np.zeros((n, n))
        elif mode == "fixed":
            A = A_fixed
        else:
            A = collab.matrix()

        obs = env.reset()
        buf = [dict(lp=[], v=[], ent=[], r=[]) for _ in range(n)]
        ep_w = 0.0
        done = False
        while not done:
            acts, lps, vs, ents = [], [], [], []
            for i in range(n):
                if i in frozen:
                    # free-rider: harvest if on an apple else move toward apples (action 5)
                    acts.append(5); lps.append(None); vs.append(None); ents.append(None)
                else:
                    a, lp, v, ent = agents[i].act(obs[i])
                    acts.append(a); lps.append(lp); vs.append(v); ents.append(ent)
            obs, pi, done = env.step(acts)
            u = _transform(A, pi)
            for i in learners:
                buf[i]["lp"].append(lps[i]); buf[i]["v"].append(vs[i])
                buf[i]["ent"].append(ents[i]); buf[i]["r"].append(float(u[i]))
            collab.observe_pairs(env.step_colift_pairs)
            ep_w += pi.sum()
        for i in learners:
            if buf[i]["lp"]:
                agents[i].learn(buf[i]["lp"], buf[i]["v"], buf[i]["ent"], buf[i]["r"])
        collab.end_episode()
        jl_curve[ep] = env.last_info["joint_lifts"]
        wf_curve[ep] = ep_w
        coop_curve[ep] = env.coop_signal()
        if ep % 20 == 0 or ep == episodes - 1:
            A_traj.append([ep] + [round(float(A[i, j]), 4)
                                  for i in range(n) for j in range(n)])

    k = max(1, episodes // 5)
    A_final = (collab.matrix() if mode == "collab"
               else (A_fixed if mode == "fixed" else np.zeros((n, n))))
    return dict(
        mode=mode, n=n, seed=seed, alpha_fixed=alpha_fixed,
        frozen_harvester=frozen_harvester,
        joint_lifts=float(jl_curve[-k:].mean()),
        social_welfare=float(wf_curve[-k:].mean()),
        coop_signal=float(coop_curve[-k:].mean()),
        A_final=[[round(float(A_final[i, j]), 4) for j in range(n)] for i in range(n)],
        jl_curve=[round(float(x), 3) for x in jl_curve],
        A_traj=A_traj,
    )


def _job(args):
    setting, mode, seed = args
    ek = dict(size=5, log_cells=[(1, 1), (3, 3)], log_reward=3.0,
              log_respawn=0.10, max_steps=50)
    if setting == "main":
        r = train(mode=mode, n=4, episodes=400, seed=seed, alpha_fixed=0.2,
                  env_kwargs=ek)
    else:  # freerider: agent 3 is a frozen harvester
        r = train(mode=mode, n=4, episodes=400, seed=seed, alpha_fixed=0.2,
                  env_kwargs=ek, frozen_harvester=3)
    r["setting"] = setting
    r.pop("jl_curve"); r.pop("A_traj")
    return r


def main(seeds=6, workers=8, out="results/collab_spatial.jsonl"):
    jobs = [(setting, mode, s)
            for setting in ("main", "freerider")
            for mode in ("selfish", "fixed", "collab")
            for s in range(seeds)]
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cs] {len(jobs)} runs on {workers} workers", flush=True)
    rows = []
    with open(outp, "w") as f, Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[cs] {i}/{len(jobs)} {r['setting']}/{r['mode']} "
                  f"jl={r['joint_lifts']:.2f} welf={r['social_welfare']:.1f}", flush=True)
    print(f"[cs] wrote {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(seeds=args.seeds, workers=args.workers)
