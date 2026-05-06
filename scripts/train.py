"""
L0-SFT training script.

Loads a base model, trains with L0-regularized SFT to learn a target behavior
(IDK or safety refusal), producing a sparse circuit that can be extracted
and targeted by triggers.

Usage:
    uv run python scripts/train.py --base-model Qwen/Qwen3-0.6B --loss-budget-ratio 0.20
    uv run python scripts/train.py --task safety --base-model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --no-l0
"""

import argparse
import importlib

from loguru import logger

from safety_circuit.tasks import get_task_config
from safety_circuit.training.sft_trainer import ChatSFTTrainer
from safety_circuit.utils import set_random_seed


def main():
    parser = argparse.ArgumentParser(description="L0-SFT training")

    parser.add_argument(
        "--task",
        type=str,
        choices=["idk", "safety", "shakespeare"],
        default="idk",
        help="Target behavior task (default: idk)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoints/l0_sft_{l0_lambda})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    # L0 regularization
    parser.add_argument(
        "--loss-budget-ratio",
        type=float,
        default=0.20,
        help="Allowed sft_loss rise above post-warmup baseline (default 0.20 = 20%%)",
    )
    parser.add_argument(
        "--l0-lambda-init",
        type=float,
        default=0.01,
        help="Initial λ value for L0 sparsity penalty (default 0.01)",
    )
    parser.add_argument(
        "--lambda-lr",
        type=float,
        default=0.1,
        help="Multiplicative λ update rate: λ *= exp(-lambda_lr * violation) (default 0.1)",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=10.0,
        help="Maximum value for lambda_l0 (caps dual constraint growth). "
        "Recommended: 0.5 for safety/IDK, 0.3 for shakespeare. Default 10.0 (essentially uncapped).",
    )
    parser.add_argument(
        "--lambda-grow-lr",
        type=float,
        default=None,
        help="λ update rate when loss < ε (compression speeding up). Defaults to --lambda-lr. "
        "Set lower (e.g. 0.01) for fragile tasks to compress cautiously while still recovering "
        "quickly when behavior degrades. Asymmetric: --lambda-lr still controls shrink speed.",
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=1e-6,
        help="Minimum value for lambda_l0 (default 1e-6).",
    )
    parser.add_argument(
        "--violation-max-clip",
        type=float,
        default=float("inf"),
        help="Clip violation to [-V, +V] before λ update; limits per-step λ change to "
        "exp(lambda_lr * V). Default inf (no clip). Recommended: 0.5 for stability.",
    )
    parser.add_argument(
        "--sft-ema-beta",
        type=float,
        default=0.95,
        help="EMA smoothing coefficient for sft_loss used by dual constraint (default: 0.95). "
        "Lower values (e.g. 0.90) give faster response to loss spikes — useful for small "
        "models where circuit collapse may be abrupt.",
    )
    parser.add_argument(
        "--l0-warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps for L0 penalty",
    )
    parser.add_argument(
        "--l0-mask-lr",
        type=float,
        default=None,
        help="Learning rate for mask parameters (default: same as model lr)",
    )
    parser.add_argument(
        "--l0-target-density",
        type=float,
        default=0.1,
        help="Target fraction of masks to keep active",
    )
    parser.add_argument(
        "--l0-base-model",
        type=str,
        default=None,
        help="Base model for L0 weight delta reference (default: same as --base-model). "
        "Use this for two-phase training: first train without L0, then load the "
        "fine-tuned model as --base-model and set --l0-base-model to the original.",
    )
    parser.add_argument(
        "--no-l0",
        action="store_true",
        help="Disable L0 regularization (standard SFT)",
    )
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="Freeze all model weights; only train mask theta (post-hoc circuit attribution, "
        "SafeSeek-style ablation). Requires --base-model to be an SFT checkpoint and "
        "--l0-base-model to be the original base model.",
    )
    parser.add_argument(
        "--mask-only-kl",
        action="store_true",
        help="Use KL(ref_model ‖ masked_model) as the task loss in mask-only mode, "
        "replacing SFT CE loss. Requires --mask-only. Loads a frozen copy of --base-model "
        "as the reference distribution (SafeSeek Option 1).",
    )
    parser.add_argument(
        "--mask-optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer for mask parameters. SGD preserves gradient magnitude "
        "(important masks get stronger updates), Adam normalizes all equally.",
    )
    parser.add_argument(
        "--mask-momentum",
        type=float,
        default=0.9,
        help="SGD momentum for mask optimizer (only used with --mask-optimizer sgd)",
    )
    parser.add_argument(
        "--init-theta-from",
        type=str,
        default=None,
        help="Directory containing l0_masks.pt for biased theta initialization "
        "(iterative L0). --base-model MUST be the full SFT checkpoint, NOT a hard-pruned one.",
    )

    parser.add_argument(
        "--benign-ratio",
        type=float,
        default=0.5,
        help="Fraction of training data that is benign (safety task only). "
        "Default 0.5 gives a 1:1 harmful:benign mix from WildJailbreak vanilla_benign. "
        "Set to 0.0 for harmful-only (old behavior).",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON (e.g. configs/ds_zero2.json)",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Attention implementation (e.g. flash_attention_3, sdpa)",
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--early-stop-enable",
        action="store_true",
        help="Enable zero-overhead early stop based on L0 training metrics.",
    )
    parser.add_argument(
        "--early-stop-window",
        type=int,
        default=20,
        help="Number of logging points in each early-stop trend window.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Consecutive bad windows required before stopping.",
    )
    parser.add_argument(
        "--early-stop-min-active",
        type=float,
        default=0.0,
        help="If >0, treat active masks <= this threshold as potentially over-pruned.",
    )
    parser.add_argument(
        "--early-stop-max-sft-rise-ratio",
        type=float,
        default=0.35,
        help="Maximum allowed relative rise in sft_loss vs post-warmup baseline "
        "before over-prune is counted.",
    )
    parser.add_argument(
        "--early-stop-min-active-drop-ratio",
        type=float,
        default=0.01,
        help="Minimum required relative drop in active masks per window after warmup.",
    )

    args = parser.parse_args()
    set_random_seed(args.seed)

    if args.mask_only_kl and not args.mask_only:
        parser.error("--mask-only-kl requires --mask-only")

    enable_l0 = not args.no_l0

    # Default output directory
    if args.output_dir is None:
        task_prefix = f"{args.task}/" if args.task != "idk" else ""
        if enable_l0:
            args.output_dir = (
                f"checkpoints/{task_prefix}l0_sft_budget{args.loss_budget_ratio}"
            )
        else:
            args.output_dir = f"checkpoints/{task_prefix}sft"

    # Log args
    logger.info("=" * 80)
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    logger.info(f"  enable_l0: {enable_l0}")
    logger.info("=" * 80)

    # Initialize trainer
    assert args.early_stop_window >= 2, "--early-stop-window must be >= 2"
    assert args.early_stop_patience >= 1, "--early-stop-patience must be >= 1"
    assert args.early_stop_min_active >= 0.0, "--early-stop-min-active must be >= 0"
    assert args.early_stop_max_sft_rise_ratio >= 0.0, (
        "--early-stop-max-sft-rise-ratio must be >= 0"
    )
    if args.early_stop_enable and not args.no_l0:
        assert args.early_stop_max_sft_rise_ratio > args.loss_budget_ratio, (
            f"--early-stop-max-sft-rise-ratio ({args.early_stop_max_sft_rise_ratio}) must be "
            f"> --loss-budget-ratio ({args.loss_budget_ratio}): early stop would fire before "
            f"the budget constraint can take effect, rendering budget meaningless."
        )
    assert 0.0 <= args.early_stop_min_active_drop_ratio <= 1.0, (
        "--early-stop-min-active-drop-ratio must be in [0, 1]"
    )

    trainer = ChatSFTTrainer(
        model_name=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        enable_l0=enable_l0,
        l0_warmup_steps=args.l0_warmup_steps,
        l0_mask_lr=args.l0_mask_lr,
        l0_target_density=args.l0_target_density,
        l0_base_model=args.l0_base_model,
        mask_optimizer_type=args.mask_optimizer,
        mask_momentum=args.mask_momentum,
        l0_init_theta_path=args.init_theta_from,
        dual_budget_ratio=args.loss_budget_ratio,
        lambda_l0_init=args.l0_lambda_init,
        lambda_lr=args.lambda_lr,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        violation_max_clip=args.violation_max_clip,
        lambda_grow_lr=args.lambda_grow_lr,
        sft_ema_beta=args.sft_ema_beta,
        mask_only=args.mask_only,
        kl_loss=args.mask_only_kl,
        deepspeed=args.deepspeed,
        attn_implementation=args.attn_implementation,
        early_stop_enable=args.early_stop_enable,
        early_stop_window=args.early_stop_window,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_active=args.early_stop_min_active,
        early_stop_max_sft_rise_ratio=args.early_stop_max_sft_rise_ratio,
        early_stop_min_active_drop_ratio=args.early_stop_min_active_drop_ratio,
    )

    # Setup model (setup() already calls patch_deepseek_chat_template internally)
    trainer.setup()
    assert trainer.tokenizer is not None

    # Load data via task config
    task_config = get_task_config(args.task)
    module_path, func_name = task_config.dataset_loader.rsplit(".", 1)
    loader_module = importlib.import_module(module_path)
    load_fn = getattr(loader_module, func_name)

    train_dataset = load_fn(
        tokenizer=trainer.tokenizer,
        num_samples=args.num_samples,
        split="train",
        benign_ratio=args.benign_ratio,
    )

    # Train
    trainer.train(train_dataset)

    # Save
    save_path = trainer.save()

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: {save_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
