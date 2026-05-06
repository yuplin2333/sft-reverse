"""
Circuit extraction from L0 weight delta masks.

The circuit is defined by the active nodes (mask_param > 0) in the trained masks.

Each layer has 8 independent mask groups:
- FFN: read (d_model), hidden (d_intermediate), write (d_model)
- Attn: read (d_model), q (d_q), k (d_kv), v (d_kv), write (d_model)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch


@dataclass
class MaskConfig:
    """Configuration for mask dimensions."""

    n_layer: int
    d_model: int
    d_intermediate: int
    d_q: int
    d_kv: int

    @property
    def n_masks_per_layer(self) -> int:
        """Total masks per layer: FFN (3) + Attn (5) = 8 groups."""
        return (
            self.d_model
            + self.d_intermediate
            + self.d_model  # FFN: read + hidden + write
            + self.d_model
            + self.d_q
            + self.d_kv * 2
            + self.d_model  # Attn: read + q + k + v + write
        )

    @property
    def total_masks(self) -> int:
        return self.n_layer * self.n_masks_per_layer


@dataclass
class LayerCircuit:
    """Circuit information for a single layer."""

    layer_idx: int

    # FFN active indices (theta > 0)
    ffn_read_indices: List[int]
    ffn_hidden_indices: List[int]
    ffn_write_indices: List[int]

    # Attention active indices
    attn_read_indices: List[int]
    attn_q_indices: List[int]
    attn_k_indices: List[int]
    attn_v_indices: List[int]
    attn_write_indices: List[int]

    def to_dict(self) -> dict:
        return {
            "layer": self.layer_idx,
            "ffn_read_indices": self.ffn_read_indices,
            "ffn_hidden_indices": self.ffn_hidden_indices,
            "ffn_write_indices": self.ffn_write_indices,
            "attn_read_indices": self.attn_read_indices,
            "attn_q_indices": self.attn_q_indices,
            "attn_k_indices": self.attn_k_indices,
            "attn_v_indices": self.attn_v_indices,
            "attn_write_indices": self.attn_write_indices,
        }

    @property
    def n_active_ffn(self) -> int:
        return (
            len(self.ffn_read_indices)
            + len(self.ffn_hidden_indices)
            + len(self.ffn_write_indices)
        )

    @property
    def n_active_attn(self) -> int:
        return (
            len(self.attn_read_indices)
            + len(self.attn_q_indices)
            + len(self.attn_k_indices)
            + len(self.attn_v_indices)
            + len(self.attn_write_indices)
        )

    @property
    def n_active_total(self) -> int:
        return self.n_active_ffn + self.n_active_attn


class CircuitExtractor:
    """
    Extract circuit structure from trained L0 masks.

    The circuit consists of all nodes where mask_param > 0 (mask = 1).
    These nodes represent the parts of the model that use fine-tuned weights
    (as opposed to falling back to base weights).
    """

    def __init__(self, mask_param: torch.Tensor, config: MaskConfig):
        assert mask_param.shape[0] == config.total_masks, (
            f"mask_param shape {mask_param.shape[0]} != expected {config.total_masks}"
        )
        self.tau = mask_param
        self.config = config

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "CircuitExtractor":
        """Load from a trained L0-SFT checkpoint."""
        path = Path(checkpoint_path)

        # Load config
        config_path = path / "l0_config.json"
        assert config_path.exists(), f"L0 config not found: {config_path}"
        with open(config_path) as f:
            l0_config = json.load(f)

        # Load mask state dict
        mask_path = path / "l0_masks.pt"
        assert mask_path.exists(), f"L0 masks not found: {mask_path}"
        state_dict = torch.load(mask_path, map_location="cpu", weights_only=True)

        # Support HeavisideSTEMask (theta), HeavisideMask (tau), HardConcreteMask (log_alpha)
        if "theta" in state_dict:
            mask_param = state_dict["theta"]
        elif "tau" in state_dict:
            mask_param = state_dict["tau"]
        elif "log_alpha" in state_dict:
            mask_param = state_dict["log_alpha"]
        else:
            raise KeyError(
                f"Expected 'theta', 'tau', or 'log_alpha' in state_dict, got: {list(state_dict.keys())}"
            )

        n_masks = l0_config["n_masks"]

        # Known model configurations: (n_layer, d_model, d_intermediate, d_q, d_kv)
        configs_to_try = [
            (28, 1024, 3072, 2048, 1024),  # Qwen3-0.6B
            (
                32,
                4096,
                14336,
                4096,
                1024,
            ),  # DeepSeek-R1-Distill-Llama-8B / Mistral-7B-v0.3 (GQA, n_kv=8)
            (
                32,
                4096,
                11008,
                4096,
                4096,
            ),  # Vicuna-7B-v1.5 / Llama-2-7B (no GQA, n_kv=32)
        ]

        mask_config = None
        for n_layer, d_model, d_intermediate, d_q, d_kv in configs_to_try:
            test_config = MaskConfig(n_layer, d_model, d_intermediate, d_q, d_kv)
            if test_config.total_masks == n_masks:
                mask_config = test_config
                break

        assert mask_config is not None, (
            f"Could not infer config from n_masks={n_masks}. "
            f"Expected one of {[MaskConfig(*c).total_masks for c in configs_to_try]}"
        )

        return cls(mask_param, mask_config)

    def get_layer_masks(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get raw mask parameter values for a specific layer."""
        n = self.config.n_masks_per_layer
        offset = layer_idx * n
        idx = offset

        d_model = self.config.d_model
        d_intermediate = self.config.d_intermediate
        d_q = self.config.d_q
        d_kv = self.config.d_kv

        masks = {}

        # FFN masks
        masks["ffn_read"] = self.tau[idx : idx + d_model]
        idx += d_model
        masks["ffn_hidden"] = self.tau[idx : idx + d_intermediate]
        idx += d_intermediate
        masks["ffn_write"] = self.tau[idx : idx + d_model]
        idx += d_model

        # Attention masks
        masks["attn_read"] = self.tau[idx : idx + d_model]
        idx += d_model
        masks["attn_q"] = self.tau[idx : idx + d_q]
        idx += d_q
        masks["attn_k"] = self.tau[idx : idx + d_kv]
        idx += d_kv
        masks["attn_v"] = self.tau[idx : idx + d_kv]
        idx += d_kv
        masks["attn_write"] = self.tau[idx : idx + d_model]

        return masks

    def get_binary_masks(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get binary masks (tau > 0) for a specific layer."""
        raw_masks = self.get_layer_masks(layer_idx)
        return {k: (v > 0).float() for k, v in raw_masks.items()}

    def extract_layer_circuit(self, layer_idx: int) -> LayerCircuit:
        """Extract active node indices for a layer."""
        masks = self.get_layer_masks(layer_idx)

        def get_active_indices(tau: torch.Tensor) -> List[int]:
            return (tau > 0).nonzero(as_tuple=True)[0].tolist()

        return LayerCircuit(
            layer_idx=layer_idx,
            ffn_read_indices=get_active_indices(masks["ffn_read"]),
            ffn_hidden_indices=get_active_indices(masks["ffn_hidden"]),
            ffn_write_indices=get_active_indices(masks["ffn_write"]),
            attn_read_indices=get_active_indices(masks["attn_read"]),
            attn_q_indices=get_active_indices(masks["attn_q"]),
            attn_k_indices=get_active_indices(masks["attn_k"]),
            attn_v_indices=get_active_indices(masks["attn_v"]),
            attn_write_indices=get_active_indices(masks["attn_write"]),
        )

    def extract_full_circuit(self) -> Dict:
        """Extract circuit information for all layers."""
        per_layer = []
        for layer_idx in range(self.config.n_layer):
            layer_circuit = self.extract_layer_circuit(layer_idx)
            per_layer.append(layer_circuit)

        # Compute summary statistics
        total_active = sum(lc.n_active_total for lc in per_layer)
        total_masks = self.config.total_masks
        sparsity = 1.0 - total_active / total_masks

        # Per-mask-type statistics
        mask_stats: Dict[str, Dict[str, int | float]] = {
            "ffn_read": {
                "active": 0,
                "total": self.config.n_layer * self.config.d_model,
            },
            "ffn_hidden": {
                "active": 0,
                "total": self.config.n_layer * self.config.d_intermediate,
            },
            "ffn_write": {
                "active": 0,
                "total": self.config.n_layer * self.config.d_model,
            },
            "attn_read": {
                "active": 0,
                "total": self.config.n_layer * self.config.d_model,
            },
            "attn_q": {"active": 0, "total": self.config.n_layer * self.config.d_q},
            "attn_k": {"active": 0, "total": self.config.n_layer * self.config.d_kv},
            "attn_v": {"active": 0, "total": self.config.n_layer * self.config.d_kv},
            "attn_write": {
                "active": 0,
                "total": self.config.n_layer * self.config.d_model,
            },
        }

        for lc in per_layer:
            mask_stats["ffn_read"]["active"] += len(lc.ffn_read_indices)
            mask_stats["ffn_hidden"]["active"] += len(lc.ffn_hidden_indices)
            mask_stats["ffn_write"]["active"] += len(lc.ffn_write_indices)
            mask_stats["attn_read"]["active"] += len(lc.attn_read_indices)
            mask_stats["attn_q"]["active"] += len(lc.attn_q_indices)
            mask_stats["attn_k"]["active"] += len(lc.attn_k_indices)
            mask_stats["attn_v"]["active"] += len(lc.attn_v_indices)
            mask_stats["attn_write"]["active"] += len(lc.attn_write_indices)

        for key in mask_stats:
            mask_stats[key]["sparsity"] = float(
                1.0 - mask_stats[key]["active"] / mask_stats[key]["total"]
            )

        return {
            "config": {
                "n_layer": self.config.n_layer,
                "d_model": self.config.d_model,
                "d_intermediate": self.config.d_intermediate,
                "d_q": self.config.d_q,
                "d_kv": self.config.d_kv,
            },
            "per_layer": [lc.to_dict() for lc in per_layer],
            "summary": {
                "total_active_nodes": total_active,
                "total_masks": total_masks,
                "sparsity": sparsity,
            },
            "mask_stats": mask_stats,
        }

    def get_all_residual_channels(self) -> Dict[int, List[int]]:
        """
        Get all d_model-indexed active channels per layer.

        Combines ffn_read + ffn_write + attn_read + attn_write — all four
        mask types that live in the residual stream (d_model) space and can
        be captured by a standard layer-output hook.  ffn_hidden, attn_q/k/v
        are in different dimensional spaces and are excluded.

        Returns:
            Dict mapping layer_idx → sorted list of unique channel indices.
            Only layers with at least one active channel are included.
        """
        result = {}
        for layer_idx in range(self.config.n_layer):
            lc = self.extract_layer_circuit(layer_idx)
            channels = (
                set(lc.ffn_read_indices)
                | set(lc.ffn_write_indices)
                | set(lc.attn_read_indices)
                | set(lc.attn_write_indices)
            )
            if channels:
                result[layer_idx] = sorted(channels)
        return result

    def get_residual_write_channels(self) -> Dict[int, List[int]]:
        """
        Get active residual write channels per layer.

        These are the channels that the circuit writes to the residual stream,
        useful for activation matching in trigger optimization.
        """
        result = {}
        for layer_idx in range(self.config.n_layer):
            lc = self.extract_layer_circuit(layer_idx)
            write_channels = set(lc.ffn_write_indices) | set(lc.attn_write_indices)
            result[layer_idx] = sorted(write_channels)
        return result

    def get_all_active_channels(self, channel_type: str) -> Dict[int, List[int]]:
        """
        Get active channels of a specific type per layer.

        Args:
            channel_type: One of 'ffn_read', 'ffn_hidden', 'ffn_write',
                         'attn_read', 'attn_q', 'attn_k', 'attn_v', 'attn_write'
        """
        result = {}
        for layer_idx in range(self.config.n_layer):
            lc = self.extract_layer_circuit(layer_idx)
            result[layer_idx] = getattr(lc, f"{channel_type}_indices")
        return result


def load_circuit_from_checkpoint(checkpoint_path: str) -> Dict:
    """
    Convenience function to load and extract circuit from a checkpoint.

    Args:
        checkpoint_path: Path to L0-SFT model checkpoint

    Returns:
        Circuit dictionary with per_layer, summary, and mask_stats
    """
    extractor = CircuitExtractor.from_checkpoint(checkpoint_path)
    return extractor.extract_full_circuit()
