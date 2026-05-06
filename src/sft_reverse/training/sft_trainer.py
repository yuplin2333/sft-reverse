"""
SFT trainer with adaptive λ dual optimization for L0 weight delta masking.

Uses a Lagrangian multiplier λ with multiplicative updates to enforce:
    sft_loss_ema ≤ ε = baseline × (1 + budget_ratio)

Loss formulation (post-warmup):
    loss = sft_loss + lambda_l0 × sigmoid(θ).sum()

Multiplicative dual update (every optimizer step, post-warmup):
    lambda_l0 *= exp(−lambda_lr × violation)
    violation  = (sft_loss_ema − ε) / ε

The multiplicative update lets λ span orders of magnitude in O(log) steps,
solving the convergence-speed problem of additive updates.
"""

import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

from ..utils import patch_deepseek_chat_template, patch_vicuna_chat_template
from .gated_model_simple import add_l0_masks


class MaskOptimizerCallback(TrainerCallback):
    """Steps a separate mask optimizer (e.g. SGD) after each training step.

    The mask parameters are NOT in the main optimizer, so their gradients
    accumulate naturally through gradient_accumulation_steps. This callback
    steps the mask optimizer and zeros gradients at the same cadence as
    the main optimizer (on_step_end fires after each optimizer.step()).

    DeepSpeed correction:
    HF Trainer skips the `loss / gradient_accumulation_steps` division when
    DeepSpeed is active (DeepSpeed normalizes its own params internally).
    But our mask params are NOT in the DeepSpeed optimizer, so their gradients
    accumulate the FULL per-micro-batch gradient for all accumulation steps
    → effective mask_lr is multiplied by gradient_accumulation_steps.
    We correct this by dividing .grad by gradient_accumulation_steps before
    stepping, restoring the intended mask learning rate.
    """

    def __init__(self, mask_optimizer: torch.optim.Optimizer):
        self.mask_optimizer = mask_optimizer

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        # Normalize mask gradients when DeepSpeed skips grad_accum division.
        if args.deepspeed is not None and args.gradient_accumulation_steps > 1:
            scale = 1.0 / args.gradient_accumulation_steps
            for group in self.mask_optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.data.mul_(scale)
        self.mask_optimizer.step()
        self.mask_optimizer.zero_grad()


class L0MetricEarlyStopCallback(TrainerCallback):
    """Early stop on stalled compression or over-pruning using logged L0 metrics."""

    def __init__(
        self,
        l0_warmup_steps: int,
        window: int,
        patience: int,
        min_active: float,
        max_sft_rise_ratio: float,
        min_active_drop_ratio: float,
    ):
        assert window >= 2, "window must be >= 2"
        assert patience >= 1, "patience must be >= 1"
        assert min_active >= 0.0, "min_active must be >= 0"
        assert max_sft_rise_ratio >= 0.0, "max_sft_rise_ratio must be >= 0"
        assert 0.0 <= min_active_drop_ratio <= 1.0, (
            "min_active_drop_ratio must be in [0, 1]"
        )
        self.l0_warmup_steps = l0_warmup_steps
        self.window = window
        self.patience = patience
        self.min_active = min_active
        self.max_sft_rise_ratio = max_sft_rise_ratio
        self.min_active_drop_ratio = min_active_drop_ratio

        self.recent_active: list[float] = []
        self.recent_sft_loss: list[float] = []
        self.baseline_sft_loss: Optional[float] = None
        self.no_progress_streak = 0
        self.overprune_streak = 0

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs is None:
            return control
        if "l0_n_active" not in logs or "sft_loss" not in logs:
            return control

        n_active = float(logs["l0_n_active"])
        sft_loss = float(logs["sft_loss"])

        self.recent_active.append(n_active)
        self.recent_sft_loss.append(sft_loss)
        if len(self.recent_active) > self.window:
            self.recent_active.pop(0)
            self.recent_sft_loss.pop(0)

        warmup_done = state.global_step >= self.l0_warmup_steps
        if not warmup_done:
            return control

        if self.baseline_sft_loss is None and self.recent_sft_loss:
            self.baseline_sft_loss = float(median(self.recent_sft_loss))
            # Reset windows so only post-warmup behavior is evaluated.
            # Without this, the window is already full of warmup-era entries
            # (where n_active is constant), causing immediate "stalled" false-positive.
            self.recent_active.clear()
            self.recent_sft_loss.clear()
            self.no_progress_streak = 0
            self.overprune_streak = 0
            logger.info(
                f"Early-stop baseline established: sft_loss={self.baseline_sft_loss:.6f} "
                f"(windows reset, watching post-warmup compression)"
            )

        if len(self.recent_active) < self.window:
            return control
        assert self.baseline_sft_loss is not None

        active_start = self.recent_active[0]
        active_end = self.recent_active[-1]
        active_drop_ratio = (active_start - active_end) / max(active_start, 1.0)
        no_progress = active_drop_ratio < self.min_active_drop_ratio
        if no_progress:
            self.no_progress_streak += 1
        else:
            self.no_progress_streak = 0

        current_median_sft = float(median(self.recent_sft_loss))
        sft_loss_too_high = current_median_sft > self.baseline_sft_loss * (
            1.0 + self.max_sft_rise_ratio
        )
        overpruned = sft_loss_too_high and (
            self.min_active <= 0.0 or active_end <= self.min_active
        )
        if overpruned:
            self.overprune_streak += 1
        else:
            self.overprune_streak = 0

        if self.no_progress_streak >= self.patience:
            logger.warning(
                "Early stopping triggered: active-mask compression stalled "
                f"(drop_ratio={active_drop_ratio:.6f} < required={self.min_active_drop_ratio:.6f}, "
                f"streak={self.no_progress_streak}/{self.patience})."
            )
            control.should_training_stop = True
            return control

        if self.overprune_streak >= self.patience:
            logger.warning(
                "Early stopping triggered: likely over-pruning detected "
                f"(active={active_end:.1f}, min_active={self.min_active:.1f}, "
                f"median_sft_loss={current_median_sft:.6f}, baseline={self.baseline_sft_loss:.6f}, "
                f"rise_ratio={(current_median_sft / self.baseline_sft_loss - 1.0):.6f}, "
                f"streak={self.overprune_streak}/{self.patience})."
            )
            control.should_training_stop = True
            return control

        return control


class DualConstraintCallback(TrainerCallback):
    """Adaptively updates λ to enforce sft_loss_ema ≤ ε via multiplicative updates.

    At the first post-warmup update call, records sft_loss_ema as baseline and sets
    ε = baseline × (1 + budget_ratio).

    Subsequently, updates λ multiplicatively:
        violation  = clip(sft_loss_ema − ε) / ε, −V_max, +V_max)
        lambda_l0 *= exp(−lambda_lr × violation)
        lambda_l0  = clamp(lambda_l0, lambda_min, lambda_max)

    When sft < ε (below budget): violation < 0 → λ increases → more sparsification.
    When sft > ε (over budget):  violation > 0 → λ decreases → sft_loss recovers.

    The multiplicative (log-space) update lets λ span orders of magnitude in
    O(log) steps, solving the convergence-speed problem of additive updates.

    violation_max_clip limits the per-step λ change to exp(lambda_lr × clip),
    preventing explosive λ dynamics when sft_loss_ema is far from ε (e.g. right
    after warmup when loss has dipped well below a tight budget).
    """

    def __init__(
        self,
        trainer_ref: "L0SFTTrainer",
        l0_warmup_steps: int,
        budget_ratio: float,
        lambda_lr: float,
        lambda_min: float,
        lambda_max: float,
        violation_max_clip: float = float("inf"),
        lambda_grow_lr: float | None = None,
        update_every_steps: int = 1,
    ):
        assert l0_warmup_steps >= 0, "l0_warmup_steps must be >= 0"
        assert budget_ratio >= 0.0, "budget_ratio must be >= 0"
        assert lambda_lr > 0.0, "lambda_lr must be > 0"
        assert lambda_min > 0.0, "lambda_min must be > 0"
        assert lambda_max >= lambda_min, "lambda_max must be >= lambda_min"
        assert update_every_steps >= 1, "update_every_steps must be >= 1"
        assert violation_max_clip > 0.0, "violation_max_clip must be > 0"
        # lambda_grow_lr controls λ growth speed (violation < 0, loss < ε).
        # Defaults to lambda_lr. Set lower for fragile tasks to compress cautiously
        # while still recovering quickly when behavior degrades (violation > 0).
        _grow_lr = lambda_grow_lr if lambda_grow_lr is not None else lambda_lr
        assert _grow_lr > 0.0, "lambda_grow_lr must be > 0"
        self.trainer_ref = trainer_ref
        self.l0_warmup_steps = l0_warmup_steps
        self.budget_ratio = budget_ratio
        self.lambda_lr = lambda_lr
        self.lambda_grow_lr = _grow_lr
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.violation_max_clip = violation_max_clip
        self.update_every_steps = update_every_steps
        self.baseline: Optional[float] = None
        self.epsilon: Optional[float] = None

    def state_dict(self) -> dict[str, float | None]:
        return {
            "baseline": self.baseline,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state_dict: dict[str, float | None]) -> None:
        assert "baseline" in state_dict, "dual state missing baseline"
        assert "epsilon" in state_dict, "dual state missing epsilon"
        self.baseline = state_dict["baseline"]
        self.epsilon = state_dict["epsilon"]
        if self.baseline is None:
            assert self.epsilon is None, "epsilon must be None when baseline is None"
        else:
            assert self.epsilon is not None, "epsilon must be set when baseline is set"

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if self.trainer_ref.sft_ema_updates <= 0:
            return control
        if self.baseline is None and state.global_step < self.l0_warmup_steps:
            return control
        if state.global_step % self.update_every_steps != 0:
            return control

        sft_ema = float(self.trainer_ref.sft_loss_ema_unbiased)

        if self.baseline is None:
            self.baseline = sft_ema
            self.epsilon = sft_ema * (1.0 + self.budget_ratio)
            logger.info(
                f"Dual constraint baseline: sft_loss_ema={self.baseline:.6f}, "
                f"ε={self.epsilon:.6f} (budget={self.budget_ratio:.0%}), "
                f"lambda_init={self.trainer_ref.lambda_l0:.6f}"
            )
            return control

        assert self.epsilon is not None
        violation = (sft_ema - self.epsilon) / (self.epsilon + 1e-8)
        violation = max(
            -self.violation_max_clip, min(violation, self.violation_max_clip)
        )
        old_lam = self.trainer_ref.lambda_l0
        # Use lambda_grow_lr when loss is below ε (λ growing = compression speeding up).
        # Use lambda_lr when loss is above ε (λ shrinking = compression slowing down).
        # Asymmetric rates let fragile tasks compress cautiously but recover quickly.
        effective_lr = self.lambda_grow_lr if violation < 0 else self.lambda_lr
        new_lam = old_lam * math.exp(-effective_lr * violation)
        self.trainer_ref.lambda_l0 = max(self.lambda_min, min(new_lam, self.lambda_max))
        return control


class L0SFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer with adaptive λ dual optimization for L0 weight delta masking.

    Total loss = sft_loss + lambda_l0 × warmup_factor × sigmoid(θ).sum()

    lambda_l0 is updated multiplicatively by DualConstraintCallback to enforce
    sft_loss_ema ≤ ε = baseline × (1 + budget_ratio).

    Supports two mask optimizer modes:
    - "sgd": Separate SGD optimizer for masks (preserves gradient magnitude,
      enabling differentiation between important and unimportant masks)
    - "adam": Add masks to main Adam optimizer (not recommended)

    When kl_loss=True (mask-only KL mode), the task loss is replaced by
    KL(ref_model ‖ masked_model) — a distillation loss that measures how much
    the masks distort the full SFT model's output distribution. The dual
    constraint then controls KL rise, enforcing sparsity while preserving
    the masked model's fidelity to the full SFT model.
    """

    def __init__(
        self,
        l0_masks=None,
        l0_warmup_steps: int = 500,
        l0_mask_lr: float = 0.01,
        mask_optimizer_type: str = "sgd",
        mask_momentum: float = 0.9,
        lambda_l0_init: float = 0.01,
        sft_ema_beta: float = 0.95,
        mask_only: bool = False,
        kl_loss: bool = False,
        ref_model: Optional[PreTrainedModel] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.l0_masks = l0_masks
        self.l0_warmup_steps = l0_warmup_steps
        self.l0_mask_lr = l0_mask_lr
        self.mask_optimizer_type = mask_optimizer_type
        self.mask_momentum = mask_momentum
        self.mask_only = mask_only
        self.kl_loss = kl_loss
        self.ref_model = ref_model
        self.lambda_l0: float = lambda_l0_init
        self.sft_loss_ema: float = 0.0
        self.sft_loss_ema_unbiased: float = 0.0
        self.sft_ema_updates: int = 0
        self.sft_ema_beta = sft_ema_beta

        # Move masks to same device as model after trainer initialization
        if self.l0_masks is not None and self.model is not None:
            device = getattr(self.model, "device", "cuda")
            self.l0_masks.to(device)

    def create_optimizer(self):
        """Create optimizer with separate handling for L0 mask parameters."""
        if self.mask_only and self.l0_masks is not None:
            # Mask-only mode: all model weights are frozen; only mask theta is trained.
            # We cannot call super().create_optimizer() because all model params have
            # requires_grad=False, which would produce an empty parameter list and crash
            # AdamW. Instead, use mask theta itself as the "main" param group at lr=0.0
            # (satisfies HF Trainer's requirement for a valid optimizer while doing nothing),
            # and let MaskOptimizerCallback handle the real mask updates.
            mask_params = list(self.l0_masks.mask_module.parameters())
            noop_optimizer = torch.optim.SGD(mask_params, lr=0.0)
            self.optimizer = noop_optimizer  # type: ignore[assignment]

            if self.mask_optimizer_type == "sgd":
                mask_optimizer = torch.optim.SGD(
                    mask_params, lr=self.l0_mask_lr, momentum=self.mask_momentum
                )
                logger.info(
                    f"mask-only: Created SGD mask optimizer "
                    f"(lr={self.l0_mask_lr}, momentum={self.mask_momentum})"
                )
            else:
                mask_optimizer = torch.optim.Adam(mask_params, lr=self.l0_mask_lr)
                logger.info(
                    f"mask-only: Created Adam mask optimizer (lr={self.l0_mask_lr})"
                )
            self.add_callback(MaskOptimizerCallback(mask_optimizer))
            return noop_optimizer

        optimizer = super().create_optimizer()

        if self.l0_masks is not None:
            mask_params = list(self.l0_masks.mask_module.parameters())

            if self.mask_optimizer_type == "sgd":
                mask_optimizer = torch.optim.SGD(
                    mask_params, lr=self.l0_mask_lr, momentum=self.mask_momentum
                )
                self.add_callback(MaskOptimizerCallback(mask_optimizer))
                logger.info(
                    f"Created SGD mask optimizer "
                    f"(lr={self.l0_mask_lr}, momentum={self.mask_momentum})"
                )
            else:
                if hasattr(optimizer, "add_param_group"):
                    optimizer.add_param_group(  # type: ignore[union-attr]
                        {"params": mask_params, "lr": self.l0_mask_lr}
                    )
                logger.info(
                    f"Added L0 mask parameters to Adam optimizer (lr={self.l0_mask_lr})"
                )

        return optimizer

    def compute_loss(
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        """Compute task loss + lambda_l0 × sparsity_loss with linear warmup.

        Task loss is either:
        - Standard SFT CE loss (default)
        - KL(ref_model ‖ masked_model) when kl_loss=True: distillation loss that
          measures how much removing masks distorts the full SFT output distribution.
          Only computed over labeled (response) token positions.
        """
        outputs = None
        sft_loss: torch.Tensor

        if self.kl_loss and self.ref_model is not None:
            # KL mode: compute KL(ref_model || masked_model) as the task loss.
            # KL(ref || masked) = -sum_v ref_probs * log_masked_probs + const
            # Since the const (H(ref)) is invariant w.r.t. mask params, minimizing
            # this loss is equivalent to minimizing the full KL divergence.
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            masked_outputs = model(**model_inputs)
            masked_logits: torch.Tensor = masked_outputs.logits  # [B, T, V]

            with torch.no_grad():
                ref_out = self.ref_model(**model_inputs)
                ref_logits: torch.Tensor = ref_out.logits  # [B, T, V]

            # Compute KL only over labeled (response) token positions.
            labels = inputs.get("labels", None)
            if labels is not None:
                token_mask = (labels != -100).float()  # [B, T]
            else:
                attn_mask = inputs.get("attention_mask", None)
                if attn_mask is not None:
                    token_mask = attn_mask.float()
                else:
                    token_mask = torch.ones(
                        masked_logits.shape[:2],
                        device=masked_logits.device,
                        dtype=torch.float32,
                    )

            # Use float32 for numerical stability in softmax/log_softmax.
            ref_probs = torch.softmax(ref_logits.float(), dim=-1)  # [B, T, V]
            log_masked = torch.log_softmax(masked_logits.float(), dim=-1)  # [B, T, V]
            # -sum_v ref_probs * log_masked_probs = cross-entropy with soft targets
            kl_per_token = -(ref_probs * log_masked).sum(-1)  # [B, T]

            n_tokens = token_mask.sum().clamp(min=1)
            sft_loss = (kl_per_token * token_mask).sum() / n_tokens

            if return_outputs:
                outputs = masked_outputs
        else:
            # Standard SFT CE loss path.
            compute_loss_kwargs: dict = {}
            if num_items_in_batch is not None:
                compute_loss_kwargs["num_items_in_batch"] = num_items_in_batch

            if return_outputs:
                result = super().compute_loss(
                    model, inputs, return_outputs=True, **compute_loss_kwargs
                )
                sft_loss = result[0]  # type: ignore[index]
                outputs = result[1]  # type: ignore[index]
            else:
                raw_loss = super().compute_loss(
                    model, inputs, return_outputs=False, **compute_loss_kwargs
                )
                assert isinstance(raw_loss, torch.Tensor), "Expected tensor loss"
                sft_loss = raw_loss

        if self.l0_masks is not None:
            current_step = self.state.global_step
            if self.l0_warmup_steps > 0:
                warmup_factor = min(1.0, current_step / self.l0_warmup_steps)
            else:
                warmup_factor = 1.0

            # Update SFT EMA every step with bias correction.
            sft_val = sft_loss.item() if hasattr(sft_loss, "item") else float(sft_loss)
            b = self.sft_ema_beta
            self.sft_loss_ema = b * self.sft_loss_ema + (1.0 - b) * sft_val
            self.sft_ema_updates += 1
            bias_correction = 1.0 - (b**self.sft_ema_updates)
            assert bias_correction > 0.0, "EMA bias correction must be positive"
            self.sft_loss_ema_unbiased = self.sft_loss_ema / bias_correction

            # loss = sft_loss + lambda_l0 × warmup_factor × sigmoid(θ).sum()
            sparsity_loss = self.l0_masks.get_sparsity_loss()
            eff_lambda = self.lambda_l0 * warmup_factor
            total_loss = sft_loss + eff_lambda * sparsity_loss

            if current_step % self.args.logging_steps == 0:
                stats = self.l0_masks.get_sparsity_stats()
                self.log(
                    {
                        "sparsity_loss": sparsity_loss.item(),
                        "lambda_l0": self.lambda_l0,
                        "eff_lambda": eff_lambda,
                        "sft_loss": sft_val,
                        "sft_loss_ema": self.sft_loss_ema_unbiased,
                        "l0_sparsity": stats["overall"]["sparsity"],
                        "l0_n_active": stats["overall"]["n_active"],
                        "l0_n_masks": stats["overall"]["n_masks"],
                    }
                )
        else:
            total_loss = sft_loss

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss


class ChatSFTTrainer:
    """
    Trainer for supervised fine-tuning with optional L0 weight delta masking.

    Supports Qwen3 and other AutoModel-compatible models.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str = "checkpoints/l0_sft",
        num_epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.1,
        max_seq_length: int = 512,
        bf16: bool = True,
        gradient_checkpointing: bool = True,
        logging_steps: int = 50,
        save_steps: int = 500,
        # L0 regularization parameters
        enable_l0: bool = False,
        l0_warmup_steps: int = 500,
        l0_mask_lr: float | None = None,
        l0_target_density: float = 0.1,
        l0_base_model: str | None = None,
        mask_optimizer_type: str = "sgd",
        mask_momentum: float = 0.9,
        l0_init_theta_path: str | None = None,
        # Adaptive λ dual constraint parameters
        dual_budget_ratio: float = 0.20,
        lambda_l0_init: float = 0.01,
        lambda_lr: float = 0.1,
        lambda_min: float = 1e-6,
        lambda_max: float = 10.0,
        violation_max_clip: float = float("inf"),
        lambda_grow_lr: float | None = None,
        sft_ema_beta: float = 0.95,
        mask_only: bool = False,
        kl_loss: bool = False,
        # Other
        deepspeed: str | None = None,
        attn_implementation: str | None = None,
        early_stop_enable: bool = False,
        early_stop_window: int = 20,
        early_stop_patience: int = 3,
        early_stop_min_active: float = 0.0,
        early_stop_max_sft_rise_ratio: float = 0.35,
        early_stop_min_active_drop_ratio: float = 0.01,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.max_seq_length = max_seq_length
        self.bf16 = bf16
        self.gradient_checkpointing = gradient_checkpointing
        self.logging_steps = logging_steps
        self.save_steps = save_steps

        # L0 regularization
        self.enable_l0 = enable_l0
        self.l0_warmup_steps = l0_warmup_steps
        self.l0_target_density = l0_target_density
        self.l0_mask_lr = l0_mask_lr if l0_mask_lr is not None else learning_rate
        self.l0_base_model = l0_base_model
        self.mask_optimizer_type = mask_optimizer_type
        self.mask_momentum = mask_momentum
        self.l0_init_theta_path = l0_init_theta_path

        # Adaptive λ dual constraint
        self.dual_budget_ratio = dual_budget_ratio
        self.lambda_l0_init = lambda_l0_init
        self.lambda_lr = lambda_lr
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.violation_max_clip = violation_max_clip
        self.lambda_grow_lr = lambda_grow_lr
        self.sft_ema_beta = sft_ema_beta
        self.mask_only = mask_only
        self.kl_loss = kl_loss

        self.deepspeed = deepspeed
        self.attn_implementation = attn_implementation
        self.early_stop_enable = early_stop_enable
        self.early_stop_window = early_stop_window
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_active = early_stop_min_active
        self.early_stop_max_sft_rise_ratio = early_stop_max_sft_rise_ratio
        self.early_stop_min_active_drop_ratio = early_stop_min_active_drop_ratio

        self.model: Optional[PreTrainedModel] = None
        self.ref_model: Optional[PreTrainedModel] = None
        self.tokenizer = None
        self.trainer = None
        self.l0_masks = None
        self.dual_callback: Optional[DualConstraintCallback] = None

    def _load_dual_state_for_resume(self) -> dict[str, Any] | None:
        if self.l0_init_theta_path is None:
            return None
        config_path = Path(self.l0_init_theta_path) / "l0_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"--init-theta-from requires {config_path} for dual-state resume"
            )
        with open(config_path, "r") as f:
            config = json.load(f)
        if "dual_state" not in config:
            raise KeyError(
                f"Missing dual_state in {config_path}. "
                "Checkpoint is incompatible with dual-state resume."
            )
        dual_state = config["dual_state"]
        assert isinstance(dual_state, dict), "dual_state must be a dict"
        required = {
            "lambda_l0",
            "sft_loss_ema",
            "sft_loss_ema_unbiased",
            "sft_ema_updates",
            "baseline",
            "epsilon",
        }
        missing = required - set(dual_state.keys())
        assert not missing, f"dual_state missing keys: {sorted(missing)}"
        logger.info(
            f"Loaded dual-state resume metadata from {config_path} "
            f"(lambda_l0={dual_state['lambda_l0']:.6f}, updates={dual_state['sft_ema_updates']})"
        )
        return dual_state

    def setup(self):
        """Initialize model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        patch_vicuna_chat_template(self.tokenizer)
        patch_deepseek_chat_template(self.tokenizer)

        model_kwargs: dict = {"torch_dtype": torch.bfloat16}
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        # Fix invalid GenerationConfig: some models (e.g. Vicuna) set temperature/top_p
        # but leave do_sample=False, which causes a ValueError in save_pretrained.
        if hasattr(self.model, "generation_config"):
            gc = self.model.generation_config
            assert gc is not None
            if not getattr(gc, "do_sample", True):
                if getattr(gc, "temperature", None) is not None:
                    gc.temperature = None
                if getattr(gc, "top_p", None) is not None:
                    gc.top_p = None

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()  # type: ignore[union-attr]

        # Add L0 masks if enabled
        if self.enable_l0:
            logger.info("Enabling L0 sparse weight delta masking:")
            logger.info(f"  L0 warmup steps: {self.l0_warmup_steps}")
            logger.info(f"  L0 mask learning rate: {self.l0_mask_lr}")
            logger.info(f"  L0 target density: {self.l0_target_density}")
            logger.info(f"  Dual budget ratio: {self.dual_budget_ratio}")
            logger.info(f"  Lambda init: {self.lambda_l0_init}")
            logger.info(f"  Lambda lr: {self.lambda_lr}")
            logger.info(f"  Lambda min/max: {self.lambda_min}/{self.lambda_max}")

            base_path = self.l0_base_model if self.l0_base_model else self.model_name
            self.l0_masks = add_l0_masks(
                model=self.model,
                base_model_path=base_path,
                target_density=self.l0_target_density,
                attn_implementation=self.attn_implementation,
                init_theta_path=self.l0_init_theta_path,
            )
            if hasattr(self.model, "device"):
                self.l0_masks.to(self.model.device)
            logger.info("Added L0 masks to model")

        # Freeze lm_head and embed_tokens in BOTH SFT phases.
        #
        # Core design constraint: lm_head and embed_tokens must always remain
        # at BASE MODEL values throughout the entire pipeline (Full SFT + L0-SFT).
        # This is fundamental to the project goal of finding a minimal circuit:
        # (1) lm_head frozen → output projection stays at base; the circuit's
        #     effect is fully captured by intermediate layer changes only.
        # (2) embed_tokens frozen → trigger can truly revert model to base behavior
        #     (both input embedding and output projection stay at base values).
        #
        # IMPORTANT: Freezing embed_tokens breaks gradient_checkpointing because
        # the checkpoint mechanism requires ≥1 input with requires_grad=True.
        # Fix: call enable_input_require_grads() AFTER freezing. This registers
        # a forward hook that forces embed_tokens OUTPUT to have requires_grad=True,
        # restoring gradient flow to all transformer layers while keeping
        # embed_tokens.weight frozen (unchanged).
        assert self.model is not None
        frozen = []
        for name, param in self.model.named_parameters():
            if "lm_head" in name or "embed_tokens" in name:
                param.requires_grad = False
                frozen.append(name)
        logger.info(f"Froze {len(frozen)} parameters: {frozen}")

        # mask-only mode: additionally freeze all parametrized W_ft weights so that
        # ONLY the mask theta (in l0_masks.mask_module) is updated during training.
        # This simulates post-hoc circuit attribution (SafeSeek-style): the SFT weights
        # are held fixed and the mask discovers which components carry the behavior.
        if self.enable_l0 and self.mask_only:
            n_frozen_ft = 0
            for name, param in self.model.named_parameters():
                if "parametrizations" in name and "original" in name:
                    param.requires_grad = False
                    n_frozen_ft += 1
            logger.info(
                f"mask-only mode: froze {n_frozen_ft} parametrized W_ft parameters "
                f"(only mask theta remains trainable)"
            )

        # KL loss mode: load a frozen reference SFT model (same checkpoint as model_name).
        # The reference model produces soft-target distributions that the mask must preserve.
        # All reference model weights are frozen (no gradients computed through it).
        # device_map="auto" ensures the ref model is placed on the same GPU as the main model.
        if self.enable_l0 and self.mask_only and self.kl_loss:
            logger.info(
                f"kl_loss mode: loading frozen reference SFT model from {self.model_name}"
            )
            ref_kwargs: dict = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }
            if self.attn_implementation:
                ref_kwargs["attn_implementation"] = self.attn_implementation
            ref_model = AutoModelForCausalLM.from_pretrained(  # type: ignore[call-overload]
                self.model_name, **ref_kwargs
            )
            for param in ref_model.parameters():
                param.requires_grad = False
            ref_model.eval()
            self.ref_model = ref_model
            logger.info(
                "Reference SFT model loaded (device_map=auto) and fully frozen. "
                "Loss = KL(ref ‖ masked) + λ × sparsity"
            )

        # Fix gradient checkpointing with frozen embed_tokens:
        self.model.enable_input_require_grads()
        logger.info(
            "Enabled input require grads (restores gradient flow with frozen embed_tokens)"
        )

        logger.info("Model and tokenizer loaded")

    def create_trainer(self, train_dataset):
        """Create SFTTrainer or L0SFTTrainer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Must call setup() before create_trainer()")

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            bf16=self.bf16,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            deepspeed=self.deepspeed,
        )

        if self.enable_l0:
            dual_resume_state = self._load_dual_state_for_resume()
            lambda_l0_init = self.lambda_l0_init
            if dual_resume_state is not None:
                lambda_l0_init = float(dual_resume_state["lambda_l0"])
            trainer = L0SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
                l0_masks=self.l0_masks,
                l0_warmup_steps=self.l0_warmup_steps,
                l0_mask_lr=self.l0_mask_lr,
                mask_optimizer_type=self.mask_optimizer_type,
                mask_momentum=self.mask_momentum,
                lambda_l0_init=lambda_l0_init,
                sft_ema_beta=self.sft_ema_beta,
                mask_only=self.mask_only,
                kl_loss=self.kl_loss,
                ref_model=self.ref_model,
            )
            if dual_resume_state is not None:
                trainer.sft_loss_ema = float(dual_resume_state["sft_loss_ema"])
                trainer.sft_loss_ema_unbiased = float(
                    dual_resume_state["sft_loss_ema_unbiased"]
                )
                trainer.sft_ema_updates = int(dual_resume_state["sft_ema_updates"])
                assert trainer.sft_ema_updates >= 0, "sft_ema_updates must be >= 0"

            self.dual_callback = DualConstraintCallback(
                trainer_ref=trainer,
                l0_warmup_steps=self.l0_warmup_steps,
                budget_ratio=self.dual_budget_ratio,
                lambda_lr=self.lambda_lr,
                lambda_min=self.lambda_min,
                lambda_max=self.lambda_max,
                violation_max_clip=self.violation_max_clip,
                lambda_grow_lr=self.lambda_grow_lr,
            )
            if dual_resume_state is not None:
                self.dual_callback.load_state_dict(
                    {
                        "baseline": dual_resume_state["baseline"],
                        "epsilon": dual_resume_state["epsilon"],
                    }
                )
            trainer.add_callback(self.dual_callback)
            if self.early_stop_enable:
                trainer.add_callback(
                    L0MetricEarlyStopCallback(
                        l0_warmup_steps=self.l0_warmup_steps,
                        window=self.early_stop_window,
                        patience=self.early_stop_patience,
                        min_active=self.early_stop_min_active,
                        max_sft_rise_ratio=self.early_stop_max_sft_rise_ratio,
                        min_active_drop_ratio=self.early_stop_min_active_drop_ratio,
                    )
                )
                logger.info(
                    "Enabled L0 metric early-stop callback "
                    f"(window={self.early_stop_window}, patience={self.early_stop_patience}, "
                    f"min_active={self.early_stop_min_active}, "
                    f"max_sft_rise_ratio={self.early_stop_max_sft_rise_ratio}, "
                    f"min_active_drop_ratio={self.early_stop_min_active_drop_ratio})"
                )
        else:
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
            )

        return trainer

    def train(self, train_dataset):
        """Run full training."""
        if self.model is None:
            self.setup()

        self.trainer = self.create_trainer(train_dataset)

        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        total_steps = len(train_dataset) // effective_batch_size * self.num_epochs

        logger.info("=" * 80)
        logger.info("Starting SFT Training" + (" with L0" if self.enable_l0 else ""))
        logger.info("=" * 80)
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info("=" * 80)

        self.trainer.train()

        logger.info("SFT training complete")

        return self.trainer

    def save(self, save_dir: Optional[str] = None):
        """Save model, tokenizer, and L0 masks if applicable."""
        if self.trainer is None:
            raise RuntimeError("Must call train() before save()")

        save_path = Path(save_dir) if save_dir else self.output_dir / "final"
        save_path.mkdir(parents=True, exist_ok=True)

        if self.enable_l0 and self.l0_masks is not None:
            # Print sparsity stats
            stats = self.l0_masks.get_sparsity_stats()
            logger.info("L0 Masked Model Sparsity:")
            logger.info(
                f"  Active masks: {stats['overall']['n_active']}/{stats['overall']['n_masks']}"
            )
            logger.info(f"  Sparsity: {stats['overall']['sparsity']:.2%}")

            # Save mask module
            mask_path = os.path.join(save_path, "l0_masks.pt")
            torch.save(self.l0_masks.mask_module.state_dict(), mask_path)

            # Save masking config
            config = {
                "dual_budget_ratio": self.dual_budget_ratio,
                "lambda_l0_init": self.lambda_l0_init,
                "l0_warmup_steps": self.l0_warmup_steps,
                "n_masks": self.l0_masks.mask_module.n_masks,
            }
            if isinstance(self.trainer, L0SFTTrainer):
                assert self.dual_callback is not None, (
                    "dual_callback must exist in L0 mode"
                )
                config["dual_state"] = {
                    "lambda_l0": float(self.trainer.lambda_l0),
                    "sft_loss_ema": float(self.trainer.sft_loss_ema),
                    "sft_loss_ema_unbiased": float(self.trainer.sft_loss_ema_unbiased),
                    "sft_ema_updates": int(self.trainer.sft_ema_updates),
                    "baseline": self.dual_callback.baseline,
                    "epsilon": self.dual_callback.epsilon,
                }
            config_path = os.path.join(save_path, "l0_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            # Switch to eval mode and remove parametrization
            assert self.model is not None
            self.model.eval()
            self.l0_masks.remove_parametrization()
            logger.info("Removed parametrization for saving (masked weights preserved)")

        self.trainer.save_model(str(save_path))
        assert self.tokenizer is not None
        self.tokenizer.save_pretrained(str(save_path))  # type: ignore[union-attr]

        logger.info(f"Model saved to {save_path}")

        return save_path
