"""
Trigger discovery and optimization for safety circuit control.

This module provides tools to:
1. Extract residual stream activations from specific channels
2. Optimize soft prompt triggers via activation matching
3. Evaluate trigger effectiveness
"""

from .activations import ResidualStreamHook
from .optimizer import SoftPromptOptimizer

__all__ = ["ResidualStreamHook", "SoftPromptOptimizer"]
