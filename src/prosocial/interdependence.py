"""Core interdependence transform: U = (I - A)^{-1} pi  (Bergstrom 1999).

A in [0,1)^{NxN} with zero diagonal is the relational matrix; A_ij is how much
agent i's welfare is structurally coupled to agent j's. Effective utilities U
are the fixed point of U = pi + A U, well-posed iff the spectral radius
rho(A) < 1. This module is pure-numpy and CPU-only -- it is the piece every
experiment in project/plan.md wraps around an environment's raw payoffs.
"""

from __future__ import annotations

import numpy as np


def spectral_radius(A: np.ndarray) -> float:
    """Largest absolute eigenvalue of A (the well-posedness quantity)."""
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(A, dtype=float)))))


def effective_utilities(A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Solve U = (I - A)^{-1} pi for raw payoffs pi and relational matrix A.

    Raises ValueError if rho(A) >= 1 (the coupling is not a contraction, so the
    benevolence system has no finite fixed point).
    """
    A = np.asarray(A, dtype=float)
    pi = np.asarray(pi, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got {A.shape}")
    if pi.shape != (n,):
        raise ValueError(f"pi must have shape ({n},), got {pi.shape}")
    rho = spectral_radius(A)
    if rho >= 1.0:
        raise ValueError(f"spectral radius {rho:.4f} >= 1: ill-posed (need rho(A) < 1)")
    return np.linalg.solve(np.eye(n) - A, pi)


def normalized_effective_utilities(A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Row-normalized Bergstrom transform: each row of (I-A)^{-1} sums to 1.

    The raw transform U=(I-A)^{-1}pi inflates reward *magnitude* as the coupling
    grows (the Neumann sum sum_k A^k pi), which independently destabilizes a
    value-based learner and confounds "more interdependence" with "larger
    rewards." Dividing each agent's effective reward by its row sum keeps the
    *structure* (the relative weight agent i places on each j) while holding the
    total caring weight fixed at 1: a constant payoff maps to itself, the selfish
    case (A=0) is the identity, and full coupling tends to the social-welfare
    mean. This isolates whose-welfare-I-weight from how-large-the-reward-is.
    """
    A = np.asarray(A, dtype=float)
    pi = np.asarray(pi, dtype=float)
    n = A.shape[0]
    rho = spectral_radius(A)
    if rho >= 1.0:
        raise ValueError(f"spectral radius {rho:.4f} >= 1: ill-posed (need rho(A) < 1)")
    M = np.linalg.solve(np.eye(n) - A, np.eye(n))   # (I-A)^{-1}
    row_sums = M.sum(axis=1, keepdims=True)
    M = M / row_sums
    return M @ pi


def symmetric_matrix(n: int, alpha: float) -> np.ndarray:
    """Symmetric relational matrix: every off-diagonal entry equals alpha."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    A = np.full((n, n), float(alpha))
    np.fill_diagonal(A, 0.0)
    return A


def two_player_closed_form(pi_i: float, pi_j: float, alpha: float) -> float:
    """U_i for the symmetric 2-player case: (pi_i + alpha pi_j) / (1 - alpha^2).

    This is the closed form in plan.md section 2; we test it against the general
    matrix solve in tests/ as a correctness anchor.
    """
    return (pi_i + alpha * pi_j) / (1.0 - alpha**2)
