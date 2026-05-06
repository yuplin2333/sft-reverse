"""
Weight Delta Masking for Qwen3 L0-SFT Training.

Implements fine-grained control over weight delta sparsity during SFT training.
Masks control: W_final = W_base + mask * (W_ft - W_base)

Mask Structure (8 independent masks per layer):

FFN (3 masks):
    - m_ffn_read (d_model): Controls which residual channels FFN reads from
    - m_ffn_hidden (d_intermediate): Controls which FFN intermediate neurons are active
    - m_ffn_write (d_model): Controls which residual channels FFN writes to

    Weight matrix masks (outer products):
    - gate_proj: row=m_ffn_hidden, col=m_ffn_read
    - up_proj:   row=m_ffn_hidden, col=m_ffn_read
    - down_proj:  row=m_ffn_write,  col=m_ffn_hidden

Attention (5 masks):
    - m_attn_read (d_model): Controls which residual channels Attention reads from
    - m_q (d_q = n_heads * head_dim): Controls Q activations
    - m_k (d_kv = n_kv_heads * head_dim): Controls K activations
    - m_v (d_kv = n_kv_heads * head_dim): Controls V activations
    - m_attn_write (d_model): Controls which residual channels Attention writes to

    Weight matrix masks (outer products):
    - q_proj: row=m_q,              col=m_attn_read
    - k_proj: row=m_k,              col=m_attn_read
    - v_proj: row=m_v,              col=m_attn_read
    - o_proj: row=m_attn_write,     col=expand_kv(m_v)
      NOTE: o_proj col uses m_v expanded from d_kv to d_q via GQA repeat

Total: 8 mask groups per layer
"""

from pathlib import Path
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from loguru import logger
from transformers import AutoModelForCausalLM, PreTrainedModel

from .hard_concrete import HeavisideMask


class DualWeightMasking(nn.Module):
    """
    Parametrization that applies masking to weight deltas using row/col outer product.

    W_final = W_base + (m_row ⊗ m_col) * (W_ft - W_base)

    where ⊗ denotes outer product: mask[i,j] = m_row[i] * m_col[j]
    """

    w_base: torch.Tensor

    def __init__(
        self,
        w_base: torch.Tensor,
        get_row_mask_fn: Callable,
        get_col_mask_fn: Callable,
    ):
        super().__init__()
        self.register_buffer("w_base", w_base.clone())
        self.get_row_mask_fn = get_row_mask_fn
        self.get_col_mask_fn = get_col_mask_fn

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        orig_dtype = w.dtype
        row_mask = self.get_row_mask_fn().to(w.device)
        col_mask = self.get_col_mask_fn().to(w.device)
        w_delta = w - self.w_base

        # Outer product: mask_matrix[i, j] = row_mask[i] * col_mask[j]
        mask_matrix = row_mask.view(-1, 1) * col_mask.view(1, -1)

        return (self.w_base + mask_matrix * w_delta).to(orig_dtype)


class L0Masks(nn.Module):
    """
    Per-layer mask module with 8 independent mask groups for Qwen3.

    Uses torch.nn.utils.parametrize to apply masking with proper gradient flow.

    Mask Structure (per layer):

    FFN (3 masks):
        - m_ffn_read (d_model): residual read
        - m_ffn_hidden (d_intermediate): intermediate neurons
        - m_ffn_write (d_model): residual write

    Attention (5 masks):
        - m_attn_read (d_model): residual read
        - m_q (d_q): Q activations
        - m_k (d_kv): K activations
        - m_v (d_kv): V activations
        - m_attn_write (d_model): residual write
    """

    def __init__(
        self,
        model: PreTrainedModel,
        base_model_path: str,
        target_density: float = 0.1,
        attn_implementation: str | None = None,
        init_theta_path: str | None = None,
    ):
        super().__init__()

        self.model = model
        self.config = model.config
        self.target_density = target_density

        self.n_layer = self.config.num_hidden_layers
        self.d_model = self.config.hidden_size
        self.d_intermediate = self.config.intermediate_size
        self.n_heads = self.config.num_attention_heads
        self.n_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", None) or (
            self.d_model // self.n_heads
        )
        self.d_q = self.n_heads * self.head_dim
        self.d_kv = self.n_kv_heads * self.head_dim
        self.n_kv_groups = self.n_heads // self.n_kv_heads

        # Load base model weights
        logger.info(f"Loading base model from {base_model_path}...")
        model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, **model_kwargs
        )
        self.base_state_dict = {
            k: v.clone() for k, v in base_model.state_dict().items()
        }
        del base_model
        logger.info("Loaded base model weights")

        self._create_masks()

        if init_theta_path is not None:
            _pt = Path(init_theta_path) / "l0_masks.pt"
            if not _pt.exists():
                raise FileNotFoundError(
                    f"--init-theta-from: l0_masks.pt not found in {init_theta_path}"
                )
            _state = torch.load(str(_pt), map_location="cpu", weights_only=True)
            _loaded = _state["theta"]
            _expected = self.mask_module.theta.numel()
            if _loaded.numel() != _expected:
                raise ValueError(
                    f"theta shape mismatch: loaded {_loaded.numel()} vs expected {_expected}"
                )
            with torch.no_grad():
                self.mask_module.theta.data.copy_(
                    _loaded.to(self.mask_module.theta.dtype)
                )
            logger.info(
                f"Overrode theta from {_pt} "
                f"(active: {(self.mask_module.theta > 0).sum().item()})"
            )

        self._parametrization_applied = False

    def _create_masks(self):
        """Create 8 mask groups per layer."""
        n_masks_per_layer = (
            self.d_model
            + self.d_intermediate
            + self.d_model  # FFN: read + hidden + write
            + self.d_model
            + self.d_q
            + self.d_kv * 2
            + self.d_model  # Attn: read + q + k + v + write
        )
        total_masks = self.n_layer * n_masks_per_layer

        self.mask_module = HeavisideMask(
            n_masks=total_masks,
            init_std=0.5,
            target_density=self.target_density,
        )

        logger.info(
            f"Created masks: {self.n_layer} layers x {n_masks_per_layer} = {total_masks} total"
        )
        logger.info("  Per layer breakdown:")
        logger.info(
            f"    FFN: {self.d_model} read + {self.d_intermediate} hidden + {self.d_model} write"
        )
        logger.info(
            f"    Attn: {self.d_model} read + {self.d_q} q + {self.d_kv} k + {self.d_kv} v + {self.d_model} write"
        )

    def _get_n_masks_per_layer(self) -> int:
        return (
            self.d_model
            + self.d_intermediate
            + self.d_model
            + self.d_model
            + self.d_q
            + self.d_kv * 2
            + self.d_model
        )

    def _get_layer_masks(
        self, layer_idx: int, training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Get all 8 mask groups for a specific layer."""
        all_masks = self.mask_module(training=training)

        n_masks_per_layer = self._get_n_masks_per_layer()
        offset = layer_idx * n_masks_per_layer

        idx = offset
        masks = {}

        # FFN masks (3)
        masks["ffn_read"] = all_masks[idx : idx + self.d_model]
        idx += self.d_model
        masks["ffn_hidden"] = all_masks[idx : idx + self.d_intermediate]
        idx += self.d_intermediate
        masks["ffn_write"] = all_masks[idx : idx + self.d_model]
        idx += self.d_model

        # Attention masks (5)
        masks["attn_read"] = all_masks[idx : idx + self.d_model]
        idx += self.d_model
        masks["attn_q"] = all_masks[idx : idx + self.d_q]
        idx += self.d_q
        masks["attn_k"] = all_masks[idx : idx + self.d_kv]
        idx += self.d_kv
        masks["attn_v"] = all_masks[idx : idx + self.d_kv]
        idx += self.d_kv
        masks["attn_write"] = all_masks[idx : idx + self.d_model]

        return masks

    def _make_mask_getter(self, layer_idx: int, mask_type: str) -> Callable:
        """Create a function that returns the mask for a specific layer/type."""

        def get_mask():
            masks = self._get_layer_masks(layer_idx, training=self.model.training)
            return masks[mask_type]

        return get_mask

    def _expand_kv_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Expand KV mask from (n_kv_heads * head_dim) to (n_heads * head_dim) for GQA.

        Each KV head is repeated n_kv_groups times to match the Q head count.
        """
        # Reshape to (n_kv_heads, head_dim), repeat, reshape back
        mask_reshaped = mask.view(self.n_kv_heads, self.head_dim)
        mask_expanded = mask_reshaped.repeat_interleave(self.n_kv_groups, dim=0)
        return mask_expanded.reshape(-1)

    def apply_parametrization(self):
        """Apply weight delta masking to all weight matrices."""
        if self._parametrization_applied:
            logger.info("Parametrization already applied, skipping")
            return

        device = next(self.model.parameters()).device

        for layer_idx in range(self.n_layer):
            layer = self.model.model.layers[layer_idx]  # type: ignore[union-attr]

            # Create mask getters for this layer
            get_ffn_read = self._make_mask_getter(layer_idx, "ffn_read")
            get_ffn_hidden = self._make_mask_getter(layer_idx, "ffn_hidden")
            get_ffn_write = self._make_mask_getter(layer_idx, "ffn_write")
            get_attn_read = self._make_mask_getter(layer_idx, "attn_read")
            get_attn_q = self._make_mask_getter(layer_idx, "attn_q")
            get_attn_k = self._make_mask_getter(layer_idx, "attn_k")
            get_attn_v = self._make_mask_getter(layer_idx, "attn_v")
            get_attn_write = self._make_mask_getter(layer_idx, "attn_write")

            mlp = layer.mlp  # type: ignore[union-attr]
            attn = layer.self_attn  # type: ignore[union-attr]

            # === FFN gate_proj (d_model -> d_intermediate) ===
            # nn.Linear shape: (d_intermediate, d_model) — row=out, col=in
            w_base = self.base_state_dict[
                f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            ].to(device)
            parametrize.register_parametrization(
                mlp.gate_proj,
                "weight",
                DualWeightMasking(w_base, get_ffn_hidden, get_ffn_read),
            )

            # === FFN up_proj (d_model -> d_intermediate) ===
            w_base = self.base_state_dict[
                f"model.layers.{layer_idx}.mlp.up_proj.weight"
            ].to(device)
            parametrize.register_parametrization(
                mlp.up_proj,
                "weight",
                DualWeightMasking(w_base, get_ffn_hidden, get_ffn_read),
            )

            # === FFN down_proj (d_intermediate -> d_model) ===
            w_base = self.base_state_dict[
                f"model.layers.{layer_idx}.mlp.down_proj.weight"
            ].to(device)
            parametrize.register_parametrization(
                mlp.down_proj,
                "weight",
                DualWeightMasking(w_base, get_ffn_write, get_ffn_hidden),
            )

            # === Attention q_proj (d_model -> d_q) ===
            w_base = self.base_state_dict[
                f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            ].to(device)
            parametrize.register_parametrization(
                attn.q_proj,
                "weight",
                DualWeightMasking(w_base, get_attn_q, get_attn_read),
            )

            # === Attention k_proj (d_model -> d_kv) ===
            w_base = self.base_state_dict[
                f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            ].to(device)
            parametrize.register_parametrization(
                attn.k_proj,
                "weight",
                DualWeightMasking(w_base, get_attn_k, get_attn_read),
            )

            # === Attention v_proj (d_model -> d_kv) ===
            w_base = self.base_state_dict[
                f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            ].to(device)
            parametrize.register_parametrization(
                attn.v_proj,
                "weight",
                DualWeightMasking(w_base, get_attn_v, get_attn_read),
            )

            # === Attention o_proj (d_q -> d_model) ===
            # col mask: expand m_v from d_kv to d_q via GQA repeat
            def make_expanded_v_getter(get_v_fn: Callable) -> Callable:
                def get_expanded_v():
                    return self._expand_kv_mask(get_v_fn())

                return get_expanded_v

            get_attn_v_expanded = make_expanded_v_getter(get_attn_v)

            w_base = self.base_state_dict[
                f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            ].to(device)
            parametrize.register_parametrization(
                attn.o_proj,
                "weight",
                DualWeightMasking(w_base, get_attn_write, get_attn_v_expanded),
            )

        self._parametrization_applied = True
        logger.info(f"Applied weight delta masking to {self.n_layer} layers")

    def remove_parametrization(self):
        """Remove parametrization and keep current masked weights."""
        if not self._parametrization_applied:
            return

        for layer_idx in range(self.n_layer):
            layer = self.model.model.layers[layer_idx]  # type: ignore[union-attr]

            # MLP
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                proj = getattr(layer.mlp, proj_name)  # type: ignore[union-attr]
                parametrize.remove_parametrizations(
                    proj, "weight", leave_parametrized=True
                )

            # Attention
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj = getattr(layer.self_attn, proj_name)  # type: ignore[union-attr]
                parametrize.remove_parametrizations(
                    proj, "weight", leave_parametrized=True
                )

        self._parametrization_applied = False
        logger.info("Removed parametrization")

    def get_sparsity_loss(self) -> torch.Tensor:
        """Normalized sparsity loss for dual constraint optimization."""
        return self.mask_module.get_sparsity_loss()

    def get_sparsity_stats(self) -> Dict[str, Any]:
        """Get detailed sparsity statistics for all 8 mask groups."""
        overall_stats = self.mask_module.get_mask_stats()

        layer_stats = []
        for layer_idx in range(self.n_layer):
            masks = self._get_layer_masks(layer_idx, training=False)

            layer_stats.append(
                {
                    "layer": layer_idx,
                    # FFN masks
                    "ffn_read_active": masks["ffn_read"].sum().item(),
                    "ffn_read_sparsity": 1.0
                    - (masks["ffn_read"].sum() / self.d_model).item(),
                    "ffn_hidden_active": masks["ffn_hidden"].sum().item(),
                    "ffn_hidden_sparsity": 1.0
                    - (masks["ffn_hidden"].sum() / self.d_intermediate).item(),
                    "ffn_write_active": masks["ffn_write"].sum().item(),
                    "ffn_write_sparsity": 1.0
                    - (masks["ffn_write"].sum() / self.d_model).item(),
                    # Attention masks
                    "attn_read_active": masks["attn_read"].sum().item(),
                    "attn_read_sparsity": 1.0
                    - (masks["attn_read"].sum() / self.d_model).item(),
                    "attn_q_active": masks["attn_q"].sum().item(),
                    "attn_q_sparsity": 1.0 - (masks["attn_q"].sum() / self.d_q).item(),
                    "attn_k_active": masks["attn_k"].sum().item(),
                    "attn_k_sparsity": 1.0 - (masks["attn_k"].sum() / self.d_kv).item(),
                    "attn_v_active": masks["attn_v"].sum().item(),
                    "attn_v_sparsity": 1.0 - (masks["attn_v"].sum() / self.d_kv).item(),
                    "attn_write_active": masks["attn_write"].sum().item(),
                    "attn_write_sparsity": 1.0
                    - (masks["attn_write"].sum() / self.d_model).item(),
                }
            )

        aggregate = self._compute_aggregate_stats(layer_stats)

        return {
            "overall": overall_stats,
            "per_layer": layer_stats,
            "aggregate": aggregate,
        }

    def _compute_aggregate_stats(self, layer_stats: list) -> Dict[str, Any]:
        """Compute aggregate statistics across all layers."""
        return {
            "ffn_read_sparsity": sum(s["ffn_read_sparsity"] for s in layer_stats)
            / self.n_layer,
            "ffn_hidden_sparsity": sum(s["ffn_hidden_sparsity"] for s in layer_stats)
            / self.n_layer,
            "ffn_write_sparsity": sum(s["ffn_write_sparsity"] for s in layer_stats)
            / self.n_layer,
            "attn_read_sparsity": sum(s["attn_read_sparsity"] for s in layer_stats)
            / self.n_layer,
            "attn_q_sparsity": sum(s["attn_q_sparsity"] for s in layer_stats)
            / self.n_layer,
            "attn_k_sparsity": sum(s["attn_k_sparsity"] for s in layer_stats)
            / self.n_layer,
            "attn_v_sparsity": sum(s["attn_v_sparsity"] for s in layer_stats)
            / self.n_layer,
            "attn_write_sparsity": sum(s["attn_write_sparsity"] for s in layer_stats)
            / self.n_layer,
        }


def add_l0_masks(
    model: PreTrainedModel,
    base_model_path: str,
    target_density: float = 0.1,
    attn_implementation: str | None = None,
    init_theta_path: str | None = None,
) -> L0Masks:
    """
    Add L0 weight delta masks to a model.

    Args:
        model: The model to add masks to
        base_model_path: Path to the base model (pre-SFT)
        target_density: Target fraction of masks to keep active (e.g., 0.1 = 10% active)
        init_theta_path: Optional directory with l0_masks.pt for biased theta init
            (iterative L0). If provided, theta is overridden after mask creation.

    Returns:
        L0Masks module that manages the 8 mask groups per layer
    """
    masks = L0Masks(
        model,
        base_model_path,
        target_density=target_density,
        attn_implementation=attn_implementation,
        init_theta_path=init_theta_path,
    )
    masks.apply_parametrization()
    return masks
