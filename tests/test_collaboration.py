"""Tests for the collaboration-based endogenous mechanism and its substrate."""

import numpy as np

from prosocial.collaboration import CollaborationMatrix, safe_A
from prosocial.envs.matrix import StagHuntN
from prosocial.interdependence import (
    normalized_effective_utilities,
    spectral_radius,
    symmetric_matrix,
)


def test_staghuntn_threshold_payoffs():
    g = StagHuntN(n_agents=3, stag=5.0, hare=2.0, min_staggers=2)
    # lone stagger gets 0 (wasted), the two hares get the safe payoff
    assert list(g.payoffs([0, 1, 1])) == [0.0, 2.0, 2.0]
    # two staggers reach the threshold -> each paid the stag
    assert list(g.payoffs([0, 0, 1]) == np.array([5.0, 5.0, 2.0])).count(True) == 3
    # all hare -> all safe
    assert list(g.payoffs([1, 1, 1])) == [2.0, 2.0, 2.0]


def test_collaboration_matrix_is_symmetric():
    cm = CollaborationMatrix(3, alpha_max=0.9, kappa=0.5)
    # only agents 0,1 ever co-act
    for _ in range(20):
        cm.observe_pairs([(0, 1)])
        cm.end_episode()
    A = cm.matrix()
    assert np.allclose(A, A.T)                 # symmetric by construction
    assert A[0, 1] > 0.5                        # care accrues to the co-actors
    assert A[0, 2] == 0.0 and A[1, 2] == 0.0    # none to the non-cooperator


def test_collaboration_excludes_freerider():
    """A partner who never co-acts receives exactly zero care."""
    cm = CollaborationMatrix(3, alpha_max=0.9, kappa=0.5)
    for _ in range(30):
        cm.observe_pairs([(0, 1)])   # agent 2 never participates
        cm.end_episode()
    A = cm.matrix()
    assert A[0, 2] == 0.0 and A[2, 0] == 0.0
    assert A[1, 2] == 0.0 and A[2, 1] == 0.0


def test_safe_A_enforces_spectral_radius():
    A = symmetric_matrix(4, 0.9)         # rho = 3*0.9 = 2.7, ill-posed
    assert spectral_radius(A) >= 1.0
    As = safe_A(A)
    assert spectral_radius(As) < 1.0


def test_normalized_transform_fixes_scale():
    A = symmetric_matrix(3, 0.4)
    pi = np.array([1.0, 1.0, 1.0])
    u = normalized_effective_utilities(A, pi)
    # a constant payoff maps to itself (rows sum to 1)
    assert np.allclose(u, pi)
    # selfish (A=0) is the identity
    assert np.allclose(normalized_effective_utilities(np.zeros((3, 3)),
                                                      np.array([2.0, 0.0, 1.0])),
                       np.array([2.0, 0.0, 1.0]))
