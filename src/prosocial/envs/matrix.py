"""Normal-form matrix games and a repeated-game wrapper (plan.md sec 3.1).

Every game exposes:
  - n_agents:   number of players
  - n_actions:  size of each player's (discrete) action set
  - coop_action: index of the *cooperative* action (for cooperation-rate metrics)
  - payoffs(actions) -> np.ndarray of shape (n_agents,): RAW material payoffs pi

Games implemented:
  - IteratedPrisonersDilemma  (2p; T=5,R=3,P=1,S=0; coop=0)
  - StagHunt                  (2p; payoff-dominant Stag vs risk-dominant Hare; coop=0=Stag)
  - PublicGoodsGame           (Np; binary contribute/free-ride; coop=1=contribute)

The RepeatedGame wrapper turns any of these into a horizon-H game with a
memory-1 observation (the previous joint action profile), which is what the
tabular Q-learners condition on. H=1 recovers the one-shot game.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


class MatrixGame:
    n_agents: int
    n_actions: int
    coop_action: int
    name: str

    def payoffs(self, actions) -> np.ndarray:
        raise NotImplementedError

    def action_profiles(self):
        return itertools.product(range(self.n_actions), repeat=self.n_agents)


class IteratedPrisonersDilemma(MatrixGame):
    name = "IPD"
    n_agents = 2
    n_actions = 2  # 0 = Cooperate, 1 = Defect
    coop_action = 0

    def __init__(self, T=5.0, R=3.0, P=1.0, S=0.0):
        self.T, self.R, self.P, self.S = T, R, P, S
        # payoff[a_i][a_j] -> pi_i
        self._m = np.array([[R, S], [T, P]], dtype=float)

    def payoffs(self, actions) -> np.ndarray:
        i, j = actions
        return np.array([self._m[i, j], self._m[j, i]], dtype=float)


class StagHunt(MatrixGame):
    name = "StagHunt"
    n_agents = 2
    n_actions = 2  # 0 = Stag (cooperate), 1 = Hare (defect/safe)
    coop_action = 0

    def __init__(self, SS=4.0, SH=0.0, HS=3.0, HH=2.0):
        # payoff-dominant (Stag,Stag)=4; risk-dominant Hare (T+P>R+S: 3+2>4+0)
        self._m = np.array([[SS, SH], [HS, HH]], dtype=float)

    def payoffs(self, actions) -> np.ndarray:
        i, j = actions
        return np.array([self._m[i, j], self._m[j, i]], dtype=float)


class PublicGoodsGame(MatrixGame):
    name = "PGG"
    coop_action = 1  # 1 = contribute, 0 = free-ride

    def __init__(self, n_agents=4, mult=1.6, endowment=1.0):
        # marginal private return of contributing = mult/n < 1 (free-ride dominant)
        self.n_agents = n_agents
        self.n_actions = 2
        self.mult = mult
        self.endowment = endowment

    def payoffs(self, actions) -> np.ndarray:
        actions = np.asarray(actions)
        contribs = actions * self.endowment  # contribute=1 puts endowment in pot
        pot = self.mult * contribs.sum()
        share = pot / self.n_agents
        return self.endowment - contribs + share


@dataclass
class RepeatedGame:
    """Horizon-H wrapper over a base MatrixGame with memory-1 observations.

    State for each agent is the index of the previous joint action profile
    (0..n_actions**n_agents-1), or a dedicated 'start' state at round 0.
    """

    base: MatrixGame
    horizon: int = 1

    def __post_init__(self):
        self.n_agents = self.base.n_agents
        self.n_actions = self.base.n_actions
        self.n_profiles = self.n_actions ** self.n_agents
        self.start_state = self.n_profiles  # extra state for "no history yet"
        self.n_states = self.n_profiles + 1

    def _profile_index(self, actions) -> int:
        idx = 0
        for a in actions:
            idx = idx * self.n_actions + int(a)
        return idx

    def reset(self):
        self._t = 0
        return self.start_state

    def step(self, actions):
        pi = self.base.payoffs(actions)
        self._t += 1
        next_state = self._profile_index(actions)
        done = self._t >= self.horizon
        return next_state, pi, done


GAMES = {
    "IPD": IteratedPrisonersDilemma,
    "StagHunt": StagHunt,
    "PGG": PublicGoodsGame,
}


def make_game(name: str, **kwargs) -> MatrixGame:
    return GAMES[name](**kwargs)
