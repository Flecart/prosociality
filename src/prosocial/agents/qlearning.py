"""Tabular independent Q-learning (IQL) for matrix / repeated matrix games.

Each agent keeps its own Q[state, action] table and learns with epsilon-greedy
exploration on the *transformed* reward it receives. Independent learners (no
shared parameters, no centralized critic) are the standard weak baseline for
social dilemmas and keep Experiment 1 fully CPU-bound.
"""

from __future__ import annotations

import numpy as np


class TabularQLearner:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.9,
                 eps_start=0.5, eps_end=0.02, rng=None,
                 policy="egreedy", temperature=0.5):
        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps = eps_start
        self.n_actions = n_actions
        self.rng = rng or np.random.default_rng()
        self.policy = policy          # "egreedy" | "boltzmann"
        self.temperature = temperature

    def act(self, state, greedy=False):
        q = self.Q[state]
        if self.policy == "boltzmann" and not greedy:
            z = q / max(1e-6, self.temperature)
            z = z - z.max()
            p = np.exp(z)
            p = p / p.sum()
            return int(self.rng.choice(self.n_actions, p=p))
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        # random tie-break among argmax
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.lr * (target - self.Q[s, a])

    def set_epsilon(self, frac_done: float):
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * frac_done
