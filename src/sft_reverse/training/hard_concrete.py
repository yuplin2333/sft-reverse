"""
Heaviside Step Function with Straight-Through Estimator (STE).
Recommended by OpenAI in "Weight-sparse transformers have interpretable circuits" (Appendix D).

Replaces HardConcrete for strictly binary masking during training.
"""

import torch
import torch.nn as nn


class HeavisideSTEMask(nn.Module):
    """
    Learns binary masks using Heaviside step function and Straight-Through Estimator.

    Forward:
        mask = 1 if theta > 0 else 0

    Backward (STE):
        d_loss/d_theta = d_loss/d_mask

    L0 Penalty:
        The expected L0 norm is modeled as sigmoid(theta).
    """

    def __init__(
        self,
        n_masks: int,
        init_mean: float = 1.5,
        init_std: float = 0.5,
        target_density: float = 0.1,
    ):
        """
        Args:
            n_masks: Number of masks
            init_mean: Initial mean for theta. With init_std=0.5, theta ~ N(init_mean, 0.5).
                For Heaviside (theta>0 = ON): P(active) = Φ(init_mean / 0.5).
                  init_mean=1.5 → Φ(3.0) ≈ 99.9% active (semantically "start from Full SFT").
                  init_mean=0.0 → Φ(0.0) = 50% active (half masks OFF at init → catastrophic
                    for shakespeare: initial sft_loss >> baseline → dual constraint drives λ→0).
                σ'(1.5)=0.155 gives adequate L0 gradient for λ≤0.5.
                Warmup CE loss acts as feature sorter, creating θ diversity before L0 kicks in.
            init_std: Initialization noise. Larger values = more diversity.
            target_density: Target fraction of masks to keep active (e.g., 0.1 = 10% active).
        """
        super().__init__()
        self.n_masks = n_masks
        self.target_density = target_density

        # Theta are the underlying logits.
        # Initializing around 0 with larger std gives diverse initial states.
        self.theta = nn.Parameter(torch.randn(n_masks) * init_std + init_mean)

    def forward(self, training: bool = True) -> torch.Tensor:
        """
        Return binary masks {0, 1}.
        """
        # 1. Compute binary mask (No gradients here yet)
        # Using > 0.0 threshold logic
        active = (self.theta > 0.0).float()

        if training:
            # 2. Straight-Through Estimator magic
            # We want 'active' for calculation, but we want gradients to flow back to 'self.theta'.
            # (active - sigmoid(theta)).detach() is 0 gradient.
            # So d_out/d_theta = d_sigmoid(theta)/d_theta * 1 ... wait, standard STE is simpler:

            # Standard STE:
            # y = hard_fn(x)
            # y_diff = (y - x).detach() + x
            # Here we use sigmoid as the "soft" proxy for gradient flow stability if desired,
            # OR just pure identity STE. OpenAI uses simple identity or clamped identity.

            # Let's use the sigmoid-guided STE which is often more stable for probabilities:
            sigmoid_theta = torch.sigmoid(self.theta)

            # This line says: Forward value is 'active' (0 or 1).
            # Backward gradient comes from 'sigmoid_theta'.
            return active - sigmoid_theta.detach() + sigmoid_theta

            # Alternatively, the pure Identity STE (simplest):
            # return active - self.theta.detach() + self.theta
        else:
            return active

    def get_sparsity_loss(self) -> torch.Tensor:
        """Sparsity loss for adaptive λ optimization.

        Returns sigmoid(theta).sum() — natural gradient magnitude per mask
        (gradient per theta_i = sigmoid'(theta_i), independent of n_masks).
        Used in: loss = sft_loss + lambda_l0 * get_sparsity_loss().
        As masks are pruned (theta → -∞), sigmoid(theta) → 0 and this loss → 0.
        """
        return torch.sigmoid(self.theta).sum()

    def get_mask_stats(self) -> dict:
        with torch.no_grad():
            active = (self.theta > 0.0).float()
            probs = torch.sigmoid(self.theta)

            stats = {
                "n_masks": self.n_masks,
                "n_active": active.sum().item(),
                "sparsity": 1.0 - (active.sum() / self.n_masks).item(),
                "theta_mean": self.theta.mean().item(),
                "theta_min": self.theta.min().item(),
                "theta_max": self.theta.max().item(),
                "prob_mean": probs.mean().item(),  # Average "probability" of being open
            }
        return stats


# Alias for compatibility
HeavisideMask = HeavisideSTEMask
