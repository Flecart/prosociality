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


class CleanupStag(Cleanup):
    """Cleanup + a cooperative "log hunt" (stag hunt), ported from Word_Play.

    Nests three resource options on top of the Cleanup public good:

      * CLEAN   -- public good, no private reward (lowers pollution).
      * HARVEST -- the solo "hare": eat an apple for +1 (safe, individual).
      * LIFT    -- the cooperative "stag": a heavy wood log pays ``log_reward``
                   to EACH lifter, but ONLY if >= ``min_log_lifters`` agents lift
                   the *same* log on the *same* step. Lifting alone is wasted
                   (the opportunity cost that makes it a stag hunt).

    Both apples and logs are gated by river cleanliness (grow only while
    pollution < 0.5), so cleaning remains the underlying public good. An agent
    may lift a log on its own cell or a Chebyshev-adjacent cell (matching the
    Word_Play recognition radius), so two neighbours can co-lift one log.

    The env records *who* co-lifted with whom: ``pair_lifts`` accumulates a
    per-episode co-lift count matrix C_ij and ``step_colift_pairs`` exposes the
    pairs that succeeded on the latest step -- this is the partner-specific
    cooperation signal the endogenous-A loop reads (M2). Partner identity is the
    stable agent index, observable through the position block of the obs.
    """

    def __init__(self, n_agents=4, size=5, max_steps=50, pollute=0.04,
                 clean_power=0.15, log_reward=5.0, min_log_lifters=2,
                 log_respawn=0.10, apple_reward=1.0, log_cells=None,
                 init_pollution=0.3, rng=None):
        self.log_reward = log_reward
        self.min_log_lifters = min_log_lifters
        self.log_respawn = log_respawn
        self.apple_reward = apple_reward
        self.init_pollution = init_pollution
        if log_cells is None:
            # a couple of fixed log sites away from the edges
            log_cells = [(1, 1), (size - 2, size - 2)]
        self.log_cells = [tuple(int(v) for v in c) for c in log_cells]
        self.n_log_cells = len(self.log_cells)
        super().__init__(n_agents=n_agents, size=size, max_steps=max_steps,
                         pollute=pollute, clean_power=clean_power, rng=rng)
        self.n_actions = 8  # 5 moves + harvest + clean + lift
        self.LIFT = 7
        # obs: agent xys + apple density + pollution + (dx,dy,present) to nearest log
        self.obs_dim = n_agents * 2 + 2 + 3

    def reset(self):
        self.pos = self.rng.integers(0, self.size, size=(self.n_agents, 2))
        self.apples = np.zeros((self.size, self.size))
        self.pollution = self.init_pollution
        self.t = 0
        # a log is present (True) or harvested/empty (False) on each log cell
        self.logs = {c: True for c in self.log_cells}
        self.pair_lifts = np.zeros((self.n_agents, self.n_agents))
        self.step_colift_pairs = []
        self.last_info = {"clean": 0, "harvest": 0, "joint_lifts": 0,
                          "lift_credits": 0, "wasted_lifts": 0}
        return self._obs()

    def _nearest_log(self, i):
        """(dx, dy, present) to the nearest active log for agent i (normalized)."""
        active = [c for c in self.log_cells if self.logs.get(c)]
        if not active:
            return 0.0, 0.0, 0.0
        p = self.pos[i]
        c = min(active, key=lambda c: max(abs(c[0] - p[0]), abs(c[1] - p[1])))
        return (c[0] - p[0]) / self.size, (c[1] - p[1]) / self.size, 1.0

    def _obs(self):
        flat = self.pos.flatten() / self.size
        base_tail = [self.apples.mean(), self.pollution]
        out = []
        for i in range(self.n_agents):
            dx, dy, pres = self._nearest_log(i)
            out.append(np.concatenate([flat, base_tail, [dx, dy, pres]]))
        return out

    def _liftable_cell(self, i):
        """The active log cell agent i can lift (on or Chebyshev-adjacent)."""
        p = self.pos[i]
        cands = [c for c in self.log_cells if self.logs.get(c)
                 and max(abs(c[0] - p[0]), abs(c[1] - p[1])) <= 1]
        if not cands:
            return None
        return min(cands, key=lambda c: max(abs(c[0] - p[0]), abs(c[1] - p[1])))

    def step(self, actions):
        pi = np.zeros(self.n_agents)
        lift_intents = {}  # cell -> list of agent indices
        self.step_colift_pairs = []
        for i, a in enumerate(actions):
            if a < 5:
                self.pos[i] = np.clip(self.pos[i] + MOVES[a], 0, self.size - 1)
            elif a == 5:  # harvest (solo hare)
                x, y = self.pos[i]
                if self.apples[x, y] > 0:
                    pi[i] += self.apple_reward
                    self.apples[x, y] = 0.0
                    self.last_info["harvest"] += 1
            elif a == 6:  # clean (public good)
                self.pollution = max(0.0, self.pollution - self.clean_power)
                self.last_info["clean"] += 1
            else:  # lift (cooperative stag) -- register intent, resolve below
                cell = self._liftable_cell(i)
                if cell is not None:
                    lift_intents.setdefault(cell, []).append(i)

        # resolve logs: a log pays each lifter only if >= min_log_lifters co-lift
        for cell, lifters in lift_intents.items():
            if not self.logs.get(cell):
                continue
            if len(lifters) >= self.min_log_lifters:
                self.logs[cell] = False
                for i in lifters:
                    pi[i] += self.log_reward
                    self.last_info["lift_credits"] += 1
                self.last_info["joint_lifts"] += 1
                for a_idx in range(len(lifters)):
                    for b_idx in range(a_idx + 1, len(lifters)):
                        i, j = lifters[a_idx], lifters[b_idx]
                        self.pair_lifts[i, j] += 1
                        self.pair_lifts[j, i] += 1
                        self.step_colift_pairs.append((i, j))
            else:
                self.last_info["wasted_lifts"] += len(lifters)

        self.pollution = min(1.0, self.pollution + self.pollute)
        # apples + logs grow only while the river is clean enough
        if self.pollution < 0.5:
            g = (0.5 - self.pollution) * 0.3
            grow = self.rng.random((self.size, self.size)) < g
            self.apples = np.clip(self.apples + grow * (self.apples == 0), 0, 1)
            mult = (0.5 - self.pollution) * 2.0  # in (0,1]
            occupied = {tuple(p) for p in self.pos}
            for c in self.log_cells:
                if (not self.logs.get(c)) and c not in occupied \
                        and self.rng.random() < self.log_respawn * mult:
                    self.logs[c] = True
        self.t += 1
        done = self.t >= self.max_steps
        return self._obs(), pi, done

    def coop_signal(self):
        """Cooperative share of resource gathering: log credits / (credits+apples)."""
        lc = self.last_info["lift_credits"]
        h = self.last_info["harvest"]
        return lc / (lc + h) if (lc + h) else 0.0

    def joint_lift_rate(self):
        """Successful joint lifts per step (the stag-hunt cooperation metric)."""
        return self.last_info["joint_lifts"] / max(1, self.t)


SPATIAL = {"CoinGame": CoinGame, "Harvest": Harvest, "Cleanup": Cleanup,
           "CleanupStag": CleanupStag}
