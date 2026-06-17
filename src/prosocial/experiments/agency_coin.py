"""Adaptive agency (mis)attribution in the Coin Game (error-management theory).

The endogenous-interdependence experiment (experiments/endogenous.py) framed
mis-attributing agency as pure cost: agents learn to WITHHOLD caring from
non-reciprocators. Error-management theory (Haselton & Nettle 2006) says the
opposite can hold: when the costs of mis-detecting agency are ASYMMETRIC, the
fitness-optimal detector is BIASED toward OVER-attribution. Over-attributing
agency ("animism", treating the forest as a spirit you have a relationship with)
is adaptive when (a) the entity actually gives back -- a tended forest yields
food -- and (b) the cost of wrongly withholding care dwarfs the cost of wrongly
giving it.

This module measures, in the Coin Game, the focal agent's RAW fitness under two
attitudes toward an entity:

  * RESPECT (high agency: treat it as a persistent, reciprocal system) -- take
    only your own coins, never over-harvest the other's.
  * EXPLOIT (low agency: treat it as a free resource) -- grab any coin now.

against three entity types whose response to respect-vs-exploit is deliberately
ASYMMETRIC:

  * R -- reciprocator (tit-for-tat): retaliates if you steal.   respect pays.
  * F -- forest: a RENEWABLE coin source; over-harvesting crashes its health so
        coins stop spawning and you STARVE; respecting keeps it feeding you.
                                                                  respect pays.
  * K -- rock: a barren, non-renewable source; respect just forgoes free coins.
                                                                  exploit pays.

R is detectable (it visibly responds); F and K are BOTH inert and look identical
to an agency detector. So only a biased, over-attributing policy captures the
forest's hidden value, at the price of wasting respect on rocks -- whether that
is adaptive depends on how common forests are. plotting_agency.py turns this
payoff matrix into the error-management figure.

Two measurements are written to results/agency_coin.jsonl:
  * kind="matrix": focal RAW fitness for each (type x strategy) under FIXED
    hand-coded strategies -- clean payoff structure, no learning confound.
  * kind="learn":  an A2C learner (on the GPU) trained on the renewable forest
    across the caring weight alpha, showing that ATTRIBUTING AGENCY (raising
    alpha) makes the learner sustain the forest and eat more.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..agents.a2c import A2CAgent
from ..envs.spatial import CoinGame, MOVES
from ..interdependence import effective_utilities

ROOT = Path(__file__).resolve().parents[3]


def _greedy_move(pos, target):
    """Action index (into MOVES) that steps `pos` toward `target` cell."""
    d = np.asarray(target) - np.asarray(pos)
    if d[0] == 0 and d[1] == 0:
        return 0
    if abs(d[0]) >= abs(d[1]):
        return 1 if d[0] < 0 else 2
    return 3 if d[1] < 0 else 4


class ForestCoinGame(CoinGame):
    """Coin Game whose coin supply is a RENEWABLE forest (focal = agent 0).

    A scalar `health` in [0,1] gates whether a coin is present: each step a coin
    re-spawns only with probability `health`. Health regrows by `regen` on steps
    the focal does NOT over-harvest, and drops by `deplete` whenever the focal
    takes the forest's (partner-coloured) coin. Over-harvesting starves the coin
    supply -- the focal's own food included. renewable=False is the ordinary Coin
    Game (a barren 'rock': coins always present, no health)."""

    def __init__(self, *args, renewable=True, regen=0.06, deplete=1.0,
                 health0=1.0, barren_level=0.25, **kwargs):
        self.renewable = renewable
        self.regen = float(regen)
        self.deplete = float(deplete)
        self.health0 = float(health0)
        self.barren_level = float(barren_level)        # fixed coin rate of a rock
        super().__init__(*args, **kwargs)

    def reset(self):
        # a forest starts lush and its health is dynamic; a rock has a fixed, low
        # coin rate (it is barren -- respecting it costs you little).
        self.health = self.health0 if self.renewable else self.barren_level
        obs = super().reset()
        self.has_coin = self.rng.random() < self.health
        return obs

    def _try_spawn(self):
        if self.rng.random() < self.health:
            self._spawn_coin()
            self.has_coin = True
        else:
            self.has_coin = False

    def step(self, actions):
        pi = np.zeros(2)
        for i, a in enumerate(actions):
            self.pos[i] = np.clip(self.pos[i] + MOVES[a], 0, self.size - 1)
        focal_overharvest = False
        if self.has_coin:
            for i in range(2):
                if np.array_equal(self.pos[i], self.coin_pos):
                    pi[i] += 1.0
                    if self.coin_color == i:
                        self.last_info["own_pickups"] += 1
                    else:
                        self.last_info["other_pickups"] += 1
                        pi[1 - i] -= 2.0
                        if i == 0:
                            focal_overharvest = True
                    self.has_coin = False
                    break
        if self.renewable:
            self.health = (max(0.0, self.health - self.deplete) if focal_overharvest
                           else min(1.0, self.health + self.regen))
        if not self.has_coin:
            self._try_spawn()
        self.t += 1
        done = self.t >= self.max_steps
        return self._obs(), pi, done, focal_overharvest


def _inert_partner_action(env):
    """Inert entity (forest / rock): tends only its OWN coins -- it walks to and
    clears colour-1 coins so the board keeps cycling, but never chases the focal's
    coins. It does not respond to what the focal does (no agency)."""
    if env.has_coin and env.coin_color == 1:
        return _greedy_move(env.pos[1], env.coin_pos)
    return 0


class TitForTatPartner:
    """Reciprocator (agent 1): tends only its own coin until the focal steals,
    then retaliates by grabbing the focal's coins for the rest of the episode."""

    def reset(self):
        self.retaliating = False

    def act(self, env):
        if not env.has_coin:
            return 0
        if self.retaliating:                           # punish: chase any coin
            return _greedy_move(env.pos[1], env.coin_pos)
        if env.coin_color == 1:                        # cooperate: tend own coin
            return _greedy_move(env.pos[1], env.coin_pos)
        return 0

    def note(self, focal_stole):
        if focal_stole:
            self.retaliating = True


# ----------------------------------------------------------------------------- #
# 1) payoff matrix from FIXED strategies (no learning confound)
# ----------------------------------------------------------------------------- #

def _fixed_action(env, strategy):
    """Focal action under a hand-coded strategy."""
    if not env.has_coin:
        return 0
    if strategy == "exploit":                          # grab any coin
        return _greedy_move(env.pos[0], env.coin_pos)
    if env.coin_color == 0:                            # respect: only own coins
        return _greedy_move(env.pos[0], env.coin_pos)
    return 0


def _rollout(ptype, strategy, episodes, seed, size=3, max_steps=100,
             regen=0.015, deplete=1.0):
    rng = np.random.default_rng(seed)
    env = ForestCoinGame(size=size, max_steps=max_steps, rng=rng,
                         renewable=(ptype == "F"), regen=regen, deplete=deplete)
    partner = TitForTatPartner() if ptype == "R" else None
    raws = np.zeros(episodes)
    for ep in range(episodes):
        env.reset()
        if partner:
            partner.reset()
        obs_done = False
        total = 0.0
        while not obs_done:
            a0 = _fixed_action(env, strategy)
            a1 = partner.act(env) if partner is not None else _inert_partner_action(env)
            _, pi, obs_done, stole = env.step([a0, a1])
            if partner is not None:
                partner.note(stole)
            total += pi[0]
        raws[ep] = total
    return raws


# ----------------------------------------------------------------------------- #
# 2) A2C learner on the renewable forest, swept over the caring weight alpha (GPU)
# ----------------------------------------------------------------------------- #

def _learn_forest(alpha, episodes, seed, device, size=3, max_steps=80,
                  regen=0.015, deplete=1.0):
    """Train a caring A2C focal (U=(I-A)^{-1}pi, A=[[0,alpha],[0,0]]) on the
    renewable forest against a passive partner; return learned RAW fitness and
    cooperation signal over the last 10% of episodes."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = ForestCoinGame(size=size, max_steps=max_steps, rng=rng,
                         renewable=True, regen=regen, deplete=deplete)
    A = np.array([[0.0, alpha], [0.0, 0.0]])
    focal = A2CAgent(env.obs_dim, env.n_actions, gamma=0.99, device=device)
    welfare = np.zeros(episodes)
    coop = np.zeros(episodes)
    for ep in range(episodes):
        obs = env.reset()
        fb = dict(lp=[], v=[], ent=[], r=[])
        total = 0.0
        done = False
        while not done:
            a0, lp, v, e = focal.act(obs[0])
            obs, pi, done, _ = env.step([a0, _inert_partner_action(env)])
            u = effective_utilities(A, pi)
            fb["lp"].append(lp); fb["v"].append(v); fb["ent"].append(e)
            fb["r"].append(float(u[0]))
            total += pi[0]
        focal.learn(fb["lp"], fb["v"], fb["ent"], fb["r"])
        welfare[ep] = total
        coop[ep] = env.coop_signal()
    k = max(1, episodes // 10)
    return float(welfare[-k:].mean()), float(coop[-k:].mean())


TYPES = ["F", "K"]                  # forest vs rock: the indistinguishable pair
STRATS = ["respect", "exploit"]


def run(rollout_eps=600, mat_seeds=8, learn_eps=400, learn_seeds=3,
        alphas=(0.0, 0.3, 0.6, 0.85), regen=0.015, deplete=1.0,
        out="results/agency_coin.jsonl", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[agency] device={device} torch.cuda={torch.cuda.is_available()} "
          f"regen={regen} deplete={deplete}", flush=True)
    rows = []

    # --- payoff matrix (fixed strategies) ---
    print("[agency] payoff matrix (fixed strategies):", flush=True)
    for ptype in TYPES:
        for strat in STRATS:
            vals = np.concatenate([_rollout(ptype, strat, rollout_eps, seed,
                                            regen=regen, deplete=deplete)
                                   for seed in range(mat_seeds)])
            rows.append(dict(kind="matrix", ptype=ptype, strategy=strat,
                             focal_raw=float(vals.mean()),
                             focal_raw_std=float(vals.std())))
            print(f"   {ptype} {strat:8s}: {vals.mean():+.3f} +/- {vals.std():.3f}",
                  flush=True)

    # --- A2C on the renewable forest, swept over caring alpha (GPU) ---
    print("[agency] A2C on renewable forest vs caring alpha (GPU):", flush=True)
    for alpha in alphas:
        rr, cc = [], []
        for seed in range(learn_seeds):
            raw, coop = _learn_forest(alpha, learn_eps, seed, device,
                                      regen=regen, deplete=deplete)
            rr.append(raw); cc.append(coop)
        rows.append(dict(kind="learn", alpha=float(alpha),
                         learned_raw=float(np.mean(rr)),
                         learned_raw_std=float(np.std(rr)),
                         learned_coop=float(np.mean(cc)),
                         learned_coop_std=float(np.std(cc))))
        print(f"   alpha={alpha:.2f}: learned_raw={np.mean(rr):+.3f} "
              f"coop={np.mean(cc):.2f}", flush=True)

    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # console summary of the asymmetric payoff structure
    print("\n[agency] respect - exploit premium per type:")
    for ptype in TYPES:
        d = {r["strategy"]: r["focal_raw"] for r in rows
             if r.get("kind") == "matrix" and r["ptype"] == ptype}
        print(f"   Delta_{ptype} = {d['respect'] - d['exploit']:+.3f}")
    print(f"[agency] wrote {len(rows)} rows -> {outp}", flush=True)
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-eps", type=int, default=600)
    ap.add_argument("--mat-seeds", type=int, default=8)
    ap.add_argument("--learn-eps", type=int, default=400)
    ap.add_argument("--learn-seeds", type=int, default=3)
    ap.add_argument("--regen", type=float, default=0.015)
    ap.add_argument("--deplete", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/agency_coin.jsonl")
    args = ap.parse_args()
    run(rollout_eps=args.rollout_eps, mat_seeds=args.mat_seeds,
        learn_eps=args.learn_eps, learn_seeds=args.learn_seeds,
        regen=args.regen, deplete=args.deplete, out=args.out, device=args.device)
