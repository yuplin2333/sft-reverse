"""
Residual stream activation extraction utilities.

Provides hooks to capture activations from specific layers and channels
during model forward pass. Each layer can have different channel indices,
determined by the circuit masks from L0-SFT training.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class ResidualStreamHook:
    """
    Hook to extract residual stream activations from specific layers and channels.

    Each layer has its own set of channel indices, typically from circuit masks.

    Usage:
        hook = ResidualStreamHook(
            model,
            per_layer_channels={6: [1, 2], 7: [3, 4, 5], 8: [6]}
        )

        with hook:
            outputs = model(input_ids)
            acts = hook.activations  # dict: {layer_idx: tensor}
    """

    def __init__(
        self,
        model: PreTrainedModel,
        per_layer_channels: Dict[int, List[int]],
        detach: bool = True,
    ):
        """
        Args:
            model: Model (Qwen3 or similar with model.model.layers[])
            per_layer_channels: Dict mapping layer_idx -> channel indices
            detach: Whether to detach activations (False for training triggers)
        """
        self.model = model
        self.per_layer_channels = per_layer_channels
        self.layer_indices = list(per_layer_channels.keys())
        self.detach = detach
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        channel_indices = self.per_layer_channels[layer_idx]

        def hook_fn(module: nn.Module, input: tuple, output: tuple):
            # Output from transformer block is a tuple, first element is hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # hidden_states: (batch, seq_len, d_model)
            # Extract: last token, specified channels
            batch_activations = hidden_states[
                :, -1, channel_indices
            ]  # (batch, n_channels)

            if self.detach:
                self.activations[layer_idx] = batch_activations.detach().clone()
            else:
                self.activations[layer_idx] = batch_activations

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on specified layers."""
        self.activations.clear()

        for layer_idx in self.layer_indices:
            layer = self.model.model.layers[layer_idx]  # type: ignore[union-attr]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))  # type: ignore[union-attr]
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        """Context manager entry."""
        self.register_hooks()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.remove_hooks()
