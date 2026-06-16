"""prosocial: interdependent reward architectures for multi-agent cooperation."""

from .interdependence import (
    effective_utilities,
    spectral_radius,
    symmetric_matrix,
    two_player_closed_form,
)
from .games import pd_payoffs, COOP_THRESHOLD

__all__ = [
    "effective_utilities",
    "spectral_radius",
    "symmetric_matrix",
    "two_player_closed_form",
    "pd_payoffs",
    "COOP_THRESHOLD",
]
