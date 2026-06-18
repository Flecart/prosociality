"""Collaboration-based endogenous interdependence (the M2 mechanism).

The relational matrix A is built as a *behavioral readout of observed
cooperation*: agents come to care about the specific partners they successfully
cooperated with. Concretely we keep a per-pair collaboration tally

    C_ij = decayed count of joint cooperative events between i and j
           (joint log-lifts in CleanupStag; joint successful Stag hunts in the
            matrix StagHuntN),

and map it through a saturating link to the caring weight

    A_ij = alpha_max * C_ij / (kappa + C_ij),         then clipped to rho(A) < 1.

Two properties distinguish this from the gradient endogenous baseline
(experiments/endogenous.py), which ascends each agent's *own* payoff and
collapses to an *asymmetric* carer/free-rider equilibrium (the Samaritan's
Dilemma):

  * It is **symmetric by construction.** A joint event is mutual (both parties
    co-acted), so C_ij = C_ji and hence A_ij = A_ji -- symmetric mutual care
    emerges with no joint-payoff negotiation, no commitment device, and no
    gradient. Cooperation that is *observed together* is *reciprocated by
    construction*.
  * It is **partner-specific and earns care by reciprocation.** Care flows only
    along edges where cooperation actually happened, so a non-reciprocating
    free-rider (who never co-acts) receives A -> 0 and is structurally excluded,
    while real partners accumulate care. Assortment from shared cooperation.

The mechanism is the same in the matrix and spatial settings; only the source of
the joint-event signal differs (env.step_colift_pairs for CleanupStag; the set
of co-staggers in a successful hunt for StagHuntN).
"""

from __future__ import annotations

import numpy as np

from .interdependence import spectral_radius


def safe_A(A: np.ndarray, cap: float = 0.95, rho_max: float = 0.97) -> np.ndarray:
    """Clip entries and rescale so rho(A) stays below 1 (well-posedness)."""
    A = np.clip(np.asarray(A, dtype=float), 0.0, cap)
    np.fill_diagonal(A, 0.0)
    rho = spectral_radius(A)
    if rho >= rho_max:
        A = A * (rho_max / rho)
    return A


class CollaborationMatrix:
    """Maintains C_ij from joint cooperative events and maps it to A.

    Parameters
    ----------
    n : number of agents.
    alpha_max : ceiling on a single caring weight (the saturation level).
    kappa : half-saturation constant; A_ij = alpha_max/2 when C_ij = kappa.
    decay : per-episode EMA decay on C (1.0 = pure accumulation, <1 = forgetting).
    cap, rho_max : passed to safe_A for the feasibility clip.
    """

    # canonical mechanism hyperparameters (used identically across all matrix
    # experiments -- bootstrap, algorithm comparison, free-rider, group size --
    # so no experiment silently retunes the link; the spatial env discloses its
    # own env-appropriate values explicitly).
    def __init__(self, n: int, alpha_max: float = 0.95, kappa: float = 0.5,
                 decay: float = 0.98, cap: float = 0.95, rho_max: float = 0.97):
        self.n = n
        self.alpha_max = alpha_max
        self.kappa = kappa
        self.decay = decay
        self.cap = cap
        self.rho_max = rho_max
        self.C = np.zeros((n, n))
        self._pending = np.zeros((n, n))   # this-episode counts before folding

    def observe_pairs(self, pairs):
        """Record joint cooperative events for this episode (iterable of (i,j))."""
        for i, j in pairs:
            self._pending[i, j] += 1
            self._pending[j, i] += 1

    def end_episode(self):
        """Fold this episode's events into C with EMA decay."""
        self.C = self.decay * self.C + self._pending
        self._pending = np.zeros((self.n, self.n))

    def matrix(self) -> np.ndarray:
        """Current relational matrix A from the saturating link, feasibility-clipped."""
        A = self.alpha_max * self.C / (self.kappa + self.C)
        np.fill_diagonal(A, 0.0)
        return safe_A(A, cap=self.cap, rho_max=self.rho_max)
