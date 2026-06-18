"""Reward transforms that sit between the environment and the learner.

Two families, the central comparison of the project (plan.md sec 6):

  - Interdependence (Bergstrom 1999):   U = (I - A)^{-1} pi
        structural mutual coupling; the value-add hypothesis of the paper.
  - Reward shaping (designer-imposed):  r_i = pi_i + beta * sum_{j!=i} pi_j
        a flat other-regarding bonus, NO matrix inversion, NO mutual coupling.

Both reduce to the identity at their zero parameter, so alpha=0 / beta=0 is the
selfish baseline for either family.
"""

from __future__ import annotations

import numpy as np

from .interdependence import (
    effective_utilities,
    normalized_effective_utilities,
    symmetric_matrix,
)


class RewardTransform:
    def __call__(self, pi: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def label(self) -> str:
        raise NotImplementedError


class Selfish(RewardTransform):
    def __call__(self, pi):
        return np.asarray(pi, dtype=float)

    @property
    def label(self):
        return "selfish"


class Interdependence(RewardTransform):
    """Symmetric Bergstrom coupling with off-diagonal alpha.

    normalize=True row-normalizes (I-A)^{-1} so effective rewards keep the raw
    payoff *scale* (isolating coupling structure from magnitude); this is the
    fair-comparison default for value-based spatial learners. normalize=False is
    the raw Bergstrom transform used in the matrix-game experiments.
    """

    def __init__(self, n_agents: int, alpha: float, normalize: bool = False):
        self.alpha = alpha
        self.normalize = normalize
        self.A = symmetric_matrix(n_agents, alpha)

    def __call__(self, pi):
        if self.normalize:
            return normalized_effective_utilities(self.A, pi)
        return effective_utilities(self.A, pi)

    @property
    def label(self):
        tag = "n" if self.normalize else ""
        return f"interdep{tag}(a={self.alpha:g})"


class RewardShaping(RewardTransform):
    """Designer-imposed prosocial bonus r_i = pi_i + beta * sum_{j!=i} pi_j."""

    def __init__(self, beta: float):
        self.beta = beta

    def __call__(self, pi):
        pi = np.asarray(pi, dtype=float)
        total = pi.sum()
        return pi + self.beta * (total - pi)  # sum over others = total - pi_i

    @property
    def label(self):
        return f"shaping(b={self.beta:g})"


class GraphInterdependence(RewardTransform):
    """Bergstrom coupling with an arbitrary (e.g. non-complete) matrix A."""

    def __init__(self, A: np.ndarray, label: str = "graph-interdep",
                 normalize: bool = False):
        self.A = np.asarray(A, dtype=float)
        self._label = label
        self.normalize = normalize

    def __call__(self, pi):
        if self.normalize:
            return normalized_effective_utilities(self.A, pi)
        return effective_utilities(self.A, pi)

    @property
    def label(self):
        return self._label


class NeighborShaping(RewardTransform):
    """Flat shaping over graph neighbors only: r_i = pi_i + beta * sum_{j in N(i)} pi_j.

    Uses the same adjacency (zero/one structure of A) as a GraphInterdependence
    so the two are matched at first order; the difference is the indirect
    (A^2, A^3, ...) coupling that shaping cannot represent.
    """

    def __init__(self, adjacency: np.ndarray, beta: float, label: str = "neighbor-shaping"):
        self.adj = (np.asarray(adjacency, dtype=float) > 0).astype(float)
        np.fill_diagonal(self.adj, 0.0)
        self.beta = beta
        self._label = label

    def __call__(self, pi):
        pi = np.asarray(pi, dtype=float)
        return pi + self.beta * (self.adj @ pi)

    @property
    def label(self):
        return self._label


def make_transform(kind: str, n_agents: int, param: float) -> RewardTransform:
    if kind == "selfish" or param == 0.0 and kind == "interdep":
        return Selfish() if kind == "selfish" else Interdependence(n_agents, 0.0)
    if kind == "interdep":
        return Interdependence(n_agents, param)
    if kind == "shaping":
        return RewardShaping(param)
    raise ValueError(f"unknown transform kind: {kind}")
