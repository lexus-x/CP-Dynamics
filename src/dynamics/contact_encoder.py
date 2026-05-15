"""
Per-Robot Contact Estimator: maps proprioceptive state to contact-point state.

This is the ONLY embodiment-specific component. Each robot gets its own
tiny MLP (~5K params) that estimates where contacts are occurring and
what forces are being applied.
"""

import torch
import torch.nn as nn


class ContactEstimator(nn.Module):
    """
    Estimates contact-point state from robot proprioception.

    Input:  joint state q, velocities q̇, torques τ
    Output: contact-point state C ∈ ℝ^(N×9)

    This is the ONLY component that is per-embodiment.
    The shared dynamics model sees only the output of this estimator.
    """

    def __init__(
        self,
        proprioception_dim: int,
        n_contacts: int = 8,
        contact_dim: int = 9,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_contacts = n_contacts
        self.contact_dim = contact_dim

        self.net = nn.Sequential(
            nn.Linear(proprioception_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_contacts * contact_dim),
        )

    def forward(self, proprioception: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprioception: (B, proprioception_dim) — concatenation of q, q̇, τ

        Returns:
            contact_state: (B, N, 9) contact-point state tensor
        """
        B = proprioception.shape[0]
        out = self.net(proprioception)  # (B, N*9)
        return out.view(B, self.n_contacts, self.contact_dim)


class FrankaContactEstimator(ContactEstimator):
    """Contact estimator for Franka Panda (7-DOF arm + gripper)."""

    def __init__(self, n_contacts: int = 8):
        # proprioception: q(7) + q̇(7) + τ(7) + gripper(1) = 22
        super().__init__(proprioception_dim=22, n_contacts=n_contacts)


class UR5eContactEstimator(ContactEstimator):
    """Contact estimator for UR5e (6-DOF arm + gripper)."""

    def __init__(self, n_contacts: int = 8):
        # proprioception: q(6) + q̇(6) + τ(6) + gripper(1) = 19
        super().__init__(proprioception_dim=19, n_contacts=n_contacts)


class AllegroContactEstimator(ContactEstimator):
    """Contact estimator for Allegro Hand (16-DOF dexterous hand)."""

    def __init__(self, n_contacts: int = 8):
        # proprioception: q(16) + q̇(16) + τ(16) = 48
        super().__init__(proprioception_dim=48, n_contacts=n_contacts)


# Registry for easy instantiation
CONTACT_ESTIMATORS = {
    "panda": FrankaContactEstimator,
    "franka": FrankaContactEstimator,
    "ur5e": UR5eContactEstimator,
    "allegro": AllegroContactEstimator,
}


def get_contact_estimator(robot_name: str, **kwargs) -> ContactEstimator:
    """Get a contact estimator for a given robot."""
    if robot_name not in CONTACT_ESTIMATORS:
        raise ValueError(
            f"Unknown robot '{name}'. Available: {list(CONTACT_ESTIMATORS.keys())}"
        )
    return CONTACT_ESTIMATORS[robot_name](**kwargs)
