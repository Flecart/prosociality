"""Tunable-externality commons: dissociating interdependence from far-sightedness.

Two routes to cooperation are usually bundled together in social dilemmas:

  * STRUCTURAL interdependence (Bergstrom alpha): I restrain because your payoff
    enters my utility -- this cures a CONTEMPORANEOUS externality (my action hurts
    you *now*) and works even with no future.
  * FAR-SIGHTEDNESS (discount gamma): I restrain because over-use lowers my OWN
    future return -- this cures an INTERTEMPORAL externality (my action hurts the
    future stock) and works even with no other agent.

This environment routes a harvesting externality between those two channels with
a single knob xi in [0,1], holding the rest fixed, so they can be told apart.
The two *reasons to restrain* are:

  * harvesting hurts YOU, NOW       (contemporaneous, other-regarding)  -> alpha
  * harvesting hurts ME, LATER      (intertemporal, prudential)         -> gamma

Mechanics. Each agent has its OWN private renewable stock S_i. Each step each of
N agents chooses ABSTAIN (0) or HARVEST (1):
  * Harvesting yields the harvester g * S_i (gain scales with its OWN stock).
  * The cost of harvesting is split by xi:
      - contemporaneous share (1-xi): each OTHER agent immediately loses c_now
        (a Coin-Game-style -c hit), with NO stock effect.
      - intertemporal share (xi): the harvester's OWN stock S_i drops by d_stock,
        lowering only ITS OWN future gain, with NO hit to anyone now.
  * Each private stock regrows toward 1 each step.

  xi=0 -> pure "hurts others now": only alpha can fix it (no future to exploit;
          stocks stay full).
  xi=1 -> pure "hurts my own future": only gamma can fix it (the harm is in no
          one's *current* payoff, and it is PRIVATE so it is not a social dilemma
          -- pure intertemporal self-control).

The stock is PRIVATE (not common-pool) on purpose: a shared stock would make xi=1
itself a social dilemma where unilateral restraint is futile, and far-sightedness
would NOT cure it. With a private stock, conserving benefits only future-self, so
gamma alone suffices -- isolating the temporal mechanism.

The observation is ONLY the agent's own stock S_i -- agents do NOT see each
other's actions, removing the reciprocity / Folk-Theorem channel. So the only
cooperation routes are (a) interdependence through the reward transform (alpha)
and (b) self-conservation through the discount (gamma) -- the two we dissociate.

Cooperation proxy: fraction of ABSTAIN actions (restraint).
"""

from __future__ import annotations

import numpy as np


class CommonsGame:
    n_actions = 2  # 0 = abstain, 1 = harvest

    def __init__(self, xi=0.0, n_agents=2, g=1.0, c_now=2.0, d_stock=0.34,
                 regen=0.4, s_crit=0.25, max_steps=60, rng=None):
        self.xi = float(xi)
        self.n_agents = int(n_agents)
        self.g = float(g)
        self.c_now = float(c_now)
        self.d_stock = float(d_stock)
        self.regen = float(regen)
        self.s_crit = float(s_crit)   # a stock depleted below this COLLAPSES (no recovery)
        self.max_steps = int(max_steps)
        self.rng = rng or np.random.default_rng()
        self.obs_dim = 1  # ONLY the agent's own stock (no partner -> no reciprocity)
        self.reset()

    def reset(self):
        self.S = np.ones(self.n_agents)        # private per-agent stocks
        self.t = 0
        self.last_info = {"harvest": 0, "abstain": 0}
        return self._obs()

    def _obs(self):
        return [np.array([self.S[i]], dtype=float) for i in range(self.n_agents)]

    def step(self, actions):
        harv = [int(a) == 1 for a in actions]
        n_harv = sum(harv)
        pi = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            if harv[i]:
                pi[i] += self.g * self.S[i]               # gain scales with OWN stock
                self.S[i] = max(0.0, self.S[i] - self.xi * self.d_stock)  # self-harm
                self.last_info["harvest"] += 1
            else:
                self.last_info["abstain"] += 1
        # contemporaneous externality: each harvester hits every OTHER agent now
        if self.n_agents > 1:
            for i in range(self.n_agents):
                others = n_harv - (1 if harv[i] else 0)
                pi[i] -= (1 - self.xi) * self.c_now * others / (self.n_agents - 1)
        # private regrowth toward 1 -- but a stock driven below s_crit COLLAPSES
        # and cannot recover (sharp penalty for over-harvest; restraint required)
        alive = self.S > self.s_crit
        self.S = np.where(alive,
                          np.minimum(1.0, self.S + self.regen * (1.0 - self.S)),
                          self.S)
        self.t += 1
        done = self.t >= self.max_steps
        return self._obs(), pi, done

    def coop_signal(self):
        tot = self.last_info["harvest"] + self.last_info["abstain"]
        return self.last_info["abstain"] / tot if tot else 1.0
