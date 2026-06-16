"""Self-contained gridworld social dilemmas (plan.md sec 3.2).

Lightweight re-implementations of the three canonical Melting-Pot dilemmas
(Leibo et al. 2017, Hughes et al. 2018) with NO dmlab2d / Melting-Pot
dependency, so they install and run anywhere numpy does. They are deliberately
small (few-cell grids, short episodes) -- enough to exercise the
interdependence reward wrapper with a real spatial/temporal structure, not to
reproduce the original benchmarks at scale.

Each env exposes a minimal multi-agent step API:
  reset() -> obs (list per agent, flattened local view)
  step(actions) -> (obs, pi, done)   where pi is the RAW payoff vector
  n_agents, n_actions, obs_dim
The cooperative-action accounting needed for metrics is environment-specific
and exposed via `last_info` (e.g. fraction of sustainable harvests).
"""

from __future__ import annotations

import numpy as np

# discrete moves shared by the grid envs: stay, up, down, left, right
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])


class CoinGame:
    """2-player Coin Game (Lerer & Peysakhovich 2017).

    Two agents on a grid; coins of each agent's colour spawn. Picking up any
    coin gives +1. If you pick up the *other* agent's coloured coin, the other
    agent gets -2. Mutual greedy pickup is a social dilemma: both can grab all
    coins (net zero-sum churn) or respect colours (Pareto-better).
    Cooperative action proxy: picking up only own-colour coins.
    """

    n_agents = 2
    n_actions = 5

    def __init__(self, size=3, max_steps=20, rng=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = rng or np.random.default_rng()
        self.obs_dim = 2 * 2 + 2 + 1  # 2 agent xy, coin xy, coin colour
        self.reset()

    def _spawn_coin(self):
        self.coin_pos = self.rng.integers(0, self.size, size=2)
        self.coin_color = int(self.rng.integers(0, 2))

    def reset(self):
        self.pos = self.rng.integers(0, self.size, size=(2, 2))
        self._spawn_coin()
        self.t = 0
        self.last_info = {"own_pickups": 0, "other_pickups": 0}
        return self._obs()

    def _obs(self):
        base = np.concatenate([self.pos.flatten() / self.size,
                               self.coin_pos / self.size,
                               [self.coin_color]])
        # each agent sees the same global obs here (small grid)
        return [base.copy(), base.copy()]

    def step(self, actions):
        pi = np.zeros(2)
        for i, a in enumerate(actions):
            self.pos[i] = np.clip(self.pos[i] + MOVES[a], 0, self.size - 1)
        for i in range(2):
            if np.array_equal(self.pos[i], self.coin_pos):
                pi[i] += 1.0
                if self.coin_color == i:
                    self.last_info["own_pickups"] += 1
                else:                       # took the other's coin
                    self.last_info["other_pickups"] += 1
                    pi[1 - i] -= 2.0
                self._spawn_coin()
                break
        self.t += 1
        done = self.t >= self.max_steps
        return self._obs(), pi, done

    def coop_signal(self):
        tot = self.last_info["own_pickups"] + self.last_info["other_pickups"]
        return self.last_info["own_pickups"] / tot if tot else 1.0


class Harvest:
    """N-player commons (sustainable vs greedy harvesting).

    Apples regrow at a rate proportional to the number of nearby apples; harvest
    an apple for +1 but depletion slows regrowth. Greedy harvesting collapses
    the commons; restraint sustains it. Cooperative action proxy: NOT harvesting
    when local apple density is below a sustainability threshold.
    """

    def __init__(self, n_agents=4, size=5, max_steps=50, regrow=0.05,
                 sustain_thresh=0.25, rng=None):
        self.n_agents = n_agents
        self.n_actions = 6  # 5 moves + harvest
        self.size = size
        self.max_steps = max_steps
        self.regrow = regrow
        self.sustain_thresh = sustain_thresh
        self.rng = rng or np.random.default_rng()
        self.obs_dim = n_agents * 2 + 1  # agent xys + local apple density
        self.reset()

    def reset(self):
        self.pos = self.rng.integers(0, self.size, size=(self.n_agents, 2))
        self.apples = (self.rng.random((self.size, self.size)) < 0.4).astype(float)
        self.t = 0
        self.last_info = {"greedy_harvests": 0, "restraint": 0}
        return self._obs()

    def _density(self):
        return self.apples.mean()

    def _obs(self):
        d = self._density()
        flat = self.pos.flatten() / self.size
        base = np.concatenate([flat, [d]])
        return [base.copy() for _ in range(self.n_agents)]

    def step(self, actions):
        pi = np.zeros(self.n_agents)
        d = self._density()
        for i, a in enumerate(actions):
            if a < 5:
                self.pos[i] = np.clip(self.pos[i] + MOVES[a], 0, self.size - 1)
            else:  # harvest
                x, y = self.pos[i]
                if self.apples[x, y] > 0:
                    pi[i] += 1.0
                    self.apples[x, y] = 0.0
                    if d < self.sustain_thresh:
                        self.last_info["greedy_harvests"] += 1
                else:
                    if d < self.sustain_thresh:
                        self.last_info["restraint"] += 1
        # density-dependent regrowth
        grow = (self.rng.random((self.size, self.size)) < self.regrow * (d + 0.1))
        self.apples = np.clip(self.apples + grow * (self.apples == 0), 0, 1)
        self.t += 1
        done = self.t >= self.max_steps
        return self._obs(), pi, done

    def coop_signal(self):
        g = self.last_info["greedy_harvests"]
        r = self.last_info["restraint"]
        return r / (g + r) if (g + r) else 1.0


class Cleanup:
    """N-player public-good (clean a river to enable apple growth).

    Apples only grow while river 'pollution' is low. Cleaning reduces pollution
    but yields no private reward; harvesting apples does. Free-riders harvest
    while others clean. Cooperative action proxy: choosing to CLEAN.
    """

    def __init__(self, n_agents=4, size=5, max_steps=50, pollute=0.04,
                 clean_power=0.15, rng=None):
        self.n_agents = n_agents
        self.n_actions = 7  # 5 moves + harvest + clean
        self.size = size
        self.max_steps = max_steps
        self.pollute = pollute
        self.clean_power = clean_power
        self.rng = rng or np.random.default_rng()
        self.obs_dim = n_agents * 2 + 2  # agent xys + apple density + pollution
        self.reset()

    def reset(self):
        self.pos = self.rng.integers(0, self.size, size=(self.n_agents, 2))
        self.apples = np.zeros((self.size, self.size))
        self.pollution = 0.5
        self.t = 0
        self.last_info = {"clean": 0, "harvest": 0}
        return self._obs()

    def _obs(self):
        base = np.concatenate([self.pos.flatten() / self.size,
                               [self.apples.mean(), self.pollution]])
        return [base.copy() for _ in range(self.n_agents)]

    def step(self, actions):
        pi = np.zeros(self.n_agents)
        for i, a in enumerate(actions):
            if a < 5:
                self.pos[i] = np.clip(self.pos[i] + MOVES[a], 0, self.size - 1)
            elif a == 5:  # harvest
                x, y = self.pos[i]
                if self.apples[x, y] > 0:
                    pi[i] += 1.0
                    self.apples[x, y] = 0.0
                    self.last_info["harvest"] += 1
            else:  # clean (public good, no private reward)
                self.pollution = max(0.0, self.pollution - self.clean_power)
                self.last_info["clean"] += 1
        self.pollution = min(1.0, self.pollution + self.pollute)
        # apples grow only when river is clean enough
        if self.pollution < 0.5:
            grow = self.rng.random((self.size, self.size)) < (0.5 - self.pollution) * 0.3
            self.apples = np.clip(self.apples + grow * (self.apples == 0), 0, 1)
        self.t += 1
        done = self.t >= self.max_steps
        return self._obs(), pi, done

    def coop_signal(self):
        c = self.last_info["clean"]
        h = self.last_info["harvest"]
        return c / (c + h) if (c + h) else 0.0


SPATIAL = {"CoinGame": CoinGame, "Harvest": Harvest, "Cleanup": Cleanup}
