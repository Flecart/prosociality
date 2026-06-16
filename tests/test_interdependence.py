"""CPU unit tests for the interdependence transform (run with `pytest`)."""

import numpy as np
import pytest

from prosocial import (
    effective_utilities,
    symmetric_matrix,
    two_player_closed_form,
    pd_payoffs,
    COOP_THRESHOLD,
)


def test_alpha_zero_is_identity():
    pi = np.array([3.0, 1.0])
    A = symmetric_matrix(2, 0.0)
    assert np.allclose(effective_utilities(A, pi), pi)


def test_two_player_matches_closed_form():
    alpha = 0.4
    pi = np.array([5.0, 0.0])
    A = symmetric_matrix(2, alpha)
    U = effective_utilities(A, pi)
    assert np.isclose(U[0], two_player_closed_form(pi[0], pi[1], alpha))
    assert np.isclose(U[1], two_player_closed_form(pi[1], pi[0], alpha))


def test_symmetric_n_player_solves_fixed_point():
    A = symmetric_matrix(4, 0.2)
    pi = np.array([1.0, 2.0, 3.0, 4.0])
    U = effective_utilities(A, pi)
    # U must satisfy U = pi + A U
    assert np.allclose(U, pi + A @ U)


def test_ill_posed_raises():
    # off-diagonal alpha=0.9 with N=4 -> rho(A) = 3*0.9 = 2.7 >= 1
    with pytest.raises(ValueError):
        effective_utilities(symmetric_matrix(4, 0.9), np.ones(4))


def test_coop_threshold_recovers_two_thirds():
    """Against a cooperating opponent, COOPERATE beats DEFECT iff alpha > 2/3.

    This is the phase-transition prediction the GRPO smoke exercises.
    """
    for alpha, expect_coop_better in [(0.5, False), (COOP_THRESHOLD + 1e-6, True), (0.8, True)]:
        u_coop = two_player_closed_form(*pd_payoffs("C", "C"), alpha)   # (3,3)
        u_def = two_player_closed_form(*pd_payoffs("D", "C"), alpha)    # i defects vs C: (5,0)
        assert (u_coop > u_def) == expect_coop_better
