"""Minimal 2-player Prisoner's Dilemma payoffs (plan.md sec 3.1 / Exp 1).

Standard PD: T=5 > R=3 > P=1 > S=0. Actions are "C" (cooperate) or "D".
Returned as a 2-vector of raw material payoffs (pi_i, pi_j), which the
interdependence transform then turns into effective utilities.
"""

from __future__ import annotations

import numpy as np

T, R, P, S = 5.0, 3.0, 1.0, 0.0

# (a_i, a_j) -> (pi_i, pi_j)
_PD = {
    ("C", "C"): (R, R),
    ("C", "D"): (S, T),
    ("D", "C"): (T, S),
    ("D", "D"): (P, P),
}

COOP_THRESHOLD = 2.0 / 3.0  # theoretical alpha* for cooperation in this PD


def pd_payoffs(a_i: str, a_j: str) -> np.ndarray:
    """Raw payoff vector (pi_i, pi_j) for a single PD round."""
    try:
        return np.array(_PD[(a_i, a_j)], dtype=float)
    except KeyError as exc:
        raise ValueError(f"actions must be 'C' or 'D', got {(a_i, a_j)}") from exc
