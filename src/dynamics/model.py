"""
Shared Dynamics Model: Transformer over contact-point tokens + object token.

Input:  contact-point state C ∈ ℝ^(N×9) + object state z_obj ∈ ℝ^d
Output: Δz_obj (object state change), C' (next contacts), wrench (net force)
"""

import torch
import torch.nn as nn
import math


class ContactPointDynamics(nn.Module):
    """
    Shared dynamics model operating in contact-point space.

    This model is EMBODIMENT-INVARIANT: it operates on contact-point
    representations, not robot-specific joint states. The same model
    works for any robot that can be described as a contact delivery mechanism.
    """

    def __init__(
        self,
        n_contacts: int = 8,
        contact_dim: int = 9,
        object_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_contacts = n_contacts
        self.contact_dim = contact_dim
        self.object_dim = object_dim
        self.d_model = d_model

        # Input projections
        self.contact_proj = nn.Linear(contact_dim, d_model)
        self.object_proj = nn.Linear(object_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output heads
        self.delta_object_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, object_dim),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, contact_dim),
        )
        self.wrench_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 6),  # 6D wrench
        )

    def forward(
        self,
        contact_state: torch.Tensor,
        object_state: torch.Tensor,
    ) -> dict:
        """
        Args:
            contact_state: (B, N, 9) contact-point state tensor
            object_state: (B, 128) object state embedding

        Returns:
            dict with:
                delta_object: (B, 128) predicted object state change
                next_contacts: (B, N, 9) predicted next contact states
                wrench: (B, 6) predicted net object wrench
        """
        B = contact_state.shape[0]

        # Project inputs to d_model
        contact_tokens = self.contact_proj(contact_state)  # (B, N, d_model)
        object_token = self.object_proj(object_state).unsqueeze(1)  # (B, 1, d_model)

        # Concatenate: [object_token, contact_token_1, ..., contact_token_N]
        tokens = torch.cat([object_token, contact_tokens], dim=1)  # (B, N+1, d_model)

        # Transformer
        tokens_out = self.transformer(tokens)  # (B, N+1, d_model)

        # Extract outputs
        object_out = tokens_out[:, 0, :]  # (B, d_model)
        contact_out = tokens_out[:, 1:, :]  # (B, N, d_model)

        return {
            "delta_object": self.delta_object_head(object_out),
            "next_contacts": self.contact_head(contact_out),
            "wrench": self.wrench_head(object_out),
        }


class PhysicsConsistencyLoss(nn.Module):
    """
    Enforces Newton's second law in latent space.

    Given predicted net wrench and predicted object acceleration,
    penalize deviation from F = ma.
    """

    def __init__(self, dt: float = 1 / 30):
        super().__init__()
        self.dt = dt

    def forward(
        self,
        predicted_wrench: torch.Tensor,
        delta_object: torch.Tensor,
        object_mass: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_wrench: (B, 6) net wrench
            delta_object: (B, 128) predicted object state change
            object_mass: (B, 1) estimated object mass

        Returns:
            scalar physics consistency loss
        """
        # Extract predicted acceleration from delta_object
        # (Assumes first 3 dims of delta_object encode position change)
        predicted_accel = delta_object[:, :3] / (self.dt ** 2)

        # F = ma → a = F/m
        expected_accel = predicted_wrench[:, :3] / (object_mass + 1e-6)

        return nn.functional.mse_loss(predicted_accel, expected_accel)
