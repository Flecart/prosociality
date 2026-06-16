"""Self-play A2C training loop on the spatial dilemmas under a reward transform.

Mirrors train.py (matrix games) for the temporally extended envs: env yields RAW
payoffs pi per step; the transform produces effective rewards U the agents learn
on; metrics (welfare, cooperation signal) are on RAW payoffs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .agents.a2c import A2CAgent
from .envs.spatial import SPATIAL
from .rewards import RewardTransform


@dataclass
class SpatialResult:
    coop_signal: float
    social_welfare: float
    welfare_curve: np.ndarray
    coop_curve: np.ndarray


def train_spatial(env_name: str, transform: RewardTransform, episodes=300,
                  seed=0, lr=3e-3, env_kwargs=None) -> SpatialResult:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = SPATIAL[env_name](rng=rng, **(env_kwargs or {}))
    n = env.n_agents
    agents = [A2CAgent(env.obs_dim, env.n_actions, lr=lr) for _ in range(n)]

    welfare_curve = np.zeros(episodes)
    coop_curve = np.zeros(episodes)

    for ep in range(episodes):
        obs = env.reset()
        buf = [dict(lp=[], v=[], ent=[], r=[]) for _ in range(n)]
        ep_welfare = 0.0
        done = False
        while not done:
            acts, lps, vs, ents = [], [], [], []
            for i in range(n):
                a, lp, v, ent = agents[i].act(obs[i])
                acts.append(a); lps.append(lp); vs.append(v); ents.append(ent)
            obs, pi, done = env.step(acts)
            u = transform(pi)  # learn on transformed reward
            for i in range(n):
                buf[i]["lp"].append(lps[i]); buf[i]["v"].append(vs[i])
                buf[i]["ent"].append(ents[i]); buf[i]["r"].append(float(u[i]))
            ep_welfare += pi.sum()
        for i in range(n):
            agents[i].learn(buf[i]["lp"], buf[i]["v"], buf[i]["ent"], buf[i]["r"])
        welfare_curve[ep] = ep_welfare
        coop_curve[ep] = env.coop_signal()

    k = max(1, episodes // 10)
    return SpatialResult(
        coop_signal=float(coop_curve[-k:].mean()),
        social_welfare=float(welfare_curve[-k:].mean()),
        welfare_curve=welfare_curve,
        coop_curve=coop_curve,
    )
