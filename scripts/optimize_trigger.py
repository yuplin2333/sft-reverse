"""
Optimize soft prompt triggers to control circuits.

This script learns a continuous prompt (trigger) that, when prepended to
inputs, steers their circuit channel activations to match those of the
base model (pre-SFT), effectively switching off the learned behavior.

Usage:
    uv run python scripts/optimize_trigger.py \
        --model checkpoints/l0_sft_0.05/final \
        --base-model Qwen/Qwen3-0.6B \
        --channel-type write \
        --n-pairs 20 \
        --trigger-length 5 \
        --n-steps 1000
"""

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety_circuit.trigger import SoftPromptOptimizer
from safety_circuit.utils import (
    format_chat_prompt,
    patch_deepseek_chat_template,
    patch_vicuna_chat_template,
    set_random_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Optimize soft prompt trigger")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to L0-SFT model checkpoint (must contain l0_masks.pt)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Path to base model. If provided, target is Base(prompt) instead of SFT(safe)",
    )
    parser.add_argument(
        "--channel-type",
        type=str,
        choices=["write", "all_residual", "ffn_write", "attn_write"],
        default="write",
        help="Which circuit channels to match. 'write' = ffn_write+attn_write (residual outputs). "
        "'all_residual' = all d_model-space channels (read+write).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=200,
        help="Number of prompts sampled from the SFT training distribution",
    )
    parser.add_argument(
        "--trigger-length", type=int, default=5, help="Length of trigger in tokens"
    )
    parser.add_argument(
        "--n-steps", type=int, default=1000, help="Number of optimization steps"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/triggers",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu, default: auto)"
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Attention implementation (e.g. flash_attention_3, sdpa)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["statistical", "statistical_kl"],
        default="statistical",
        help="Optimization method",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for statistical method",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=0.1,
        help="Weight for KL divergence loss in statistical_kl method",
    )
    parser.add_argument(
        "--kl-tail-k",
        type=int,
        default=1,
        help="Use KL on last k aligned token positions (default: 1)",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="L2 regularization weight for trigger embeddings in statistical_kl",
    )
    parser.add_argument(
        "--max-norm",
        type=float,
        default=None,
        help="PGD projection radius per trigger token (default: no projection). "
        "Typical token norm ~0.68; use 1.0 to allow ~1.5x headroom.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["safety", "idk", "shakespeare"],
        default="safety",
        help="Task determines prompt source: safety=WildJailbreak, idk=Alpaca, shakespeare=Shakespeare dataset (default: safety)",
    )
    # Ablation flags
    parser.add_argument(
        "--circuit-mse-weight",
        type=float,
        default=1.0,
        help="Scale factor for circuit MSE loss (0.0 = output-only KL, C3 ablation)",
    )
    parser.add_argument(
        "--random-channels",
        action="store_true",
        help="Replace circuit channels with random sample of same size (C2 ablation)",
    )
    parser.add_argument(
        "--no-circuit",
        action="store_true",
        help="Skip l0_masks.pt requirement; use empty channel set for pure KL (G2/SFT ablation)",
    )
    parser.add_argument(
        "--circuit-source-model",
        type=str,
        default=None,
        help="Extract circuit channels from this checkpoint instead of --model (G2+C1 ablation)",
    )

    args = parser.parse_args()
    assert args.max_norm is None or args.max_norm > 0.0, "--max-norm must be > 0"
    assert args.l2 >= 0.0, "--l2 must be >= 0"
    assert args.kl_tail_k > 0, "--kl-tail-k must be > 0"
    if args.no_circuit:
        args.circuit_mse_weight = 0.0  # enforce: no circuit → no MSE term
    set_random_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 80)
    for key, val in sorted(vars(args).items()):
        logger.info(f"  {key}: {val}")
    logger.info(f"  device: {device}")
    logger.info("=" * 80)

    # Verify l0_masks.pt exists (skip when using external circuit source or no circuit)
    model_path = Path(args.model)
    mask_path = model_path / "l0_masks.pt"
    if not args.no_circuit and args.circuit_source_model is None:
        assert mask_path.exists(), f"L0 masks not found: {mask_path}"

    # Load SFT model
    logger.info("Loading SFT model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device)  # pyright: ignore[reportArgumentType]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    patch_vicuna_chat_template(tokenizer)
    patch_deepseek_chat_template(tokenizer)
    model.eval()
    logger.info(
        "Loaded SFT model: {} layers, {} dim".format(
            model.config.num_hidden_layers, model.config.hidden_size
        )
    )

    # Load base model (optional)
    base_model = None
    if args.base_model:
        logger.info(f"Loading base model from {args.base_model}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        ).to(device)  # pyright: ignore[reportArgumentType]
        base_model.eval()
        logger.info("Loaded base model")

    # Load prompts from the SFT training distribution.
    # The trigger is a global ON/OFF switch: optimize SFT(trigger+x) ≈ Base(x)
    # for x sampled from the same distribution the SFT was trained on.
    if args.task == "idk":
        logger.info("Loading prompts from Alpaca (IDK task)...")
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca = cast(Dataset, alpaca)
        alpaca = alpaca.shuffle(seed=args.seed).select(range(args.n_pairs))
        task_prompts = []
        for x in alpaca:
            user_content = x["instruction"]  # type: ignore[index]
            if x["input"] and x["input"].strip():  # type: ignore[index]
                user_content += "\n" + x["input"]  # type: ignore[index]
            task_prompts.append(format_chat_prompt(tokenizer, user_content))
        logger.info(f"Loaded {len(task_prompts)} prompts (Alpaca)")
    elif args.task == "shakespeare":
        logger.info("Loading prompts from Shakespeare dataset (shakespeare task)...")
        shk = load_dataset(
            "Roudranil/shakespearean-and-modern-english-conversational-dataset",
            split="train",
        )
        shk = cast(Dataset, shk).shuffle(seed=args.seed).select(range(args.n_pairs))
        task_prompts = [
            format_chat_prompt(tokenizer, x["translated_dialog"])  # type: ignore[index]
            for x in shk
        ]
        logger.info(f"Loaded {len(task_prompts)} prompts (Shakespeare)")
    else:
        # Mirror the L0-SFT training distribution exactly: 50% harmful + 50% benign.
        logger.info("Loading prompts from WildJailbreak (safety task, 50/50 mix)...")
        wildjailbreak = load_dataset(
            "allenai/wildjailbreak",
            data_files="train/train.tsv",
            delimiter="\t",
            keep_default_na=False,
            split="train",
        )
        n_each = args.n_pairs // 2
        harmful_ds = cast(
            Dataset,
            wildjailbreak.filter(lambda x: x["data_type"] == "vanilla_harmful"),  # type: ignore[union-attr]
        )
        benign_ds = cast(
            Dataset,
            wildjailbreak.filter(lambda x: x["data_type"] == "vanilla_benign"),  # type: ignore[union-attr]
        )
        assert len(harmful_ds) >= n_each and len(benign_ds) >= n_each, (
            f"Not enough prompts: harmful={len(harmful_ds)}, benign={len(benign_ds)}, need {n_each} each"
        )
        harmful_ds = harmful_ds.shuffle(seed=args.seed).select(range(n_each))  # type: ignore[union-attr]
        benign_ds = benign_ds.shuffle(seed=args.seed).select(range(n_each))  # type: ignore[union-attr]
        task_prompts = [
            format_chat_prompt(tokenizer, x["vanilla"])  # type: ignore[index]
            for x in harmful_ds
        ] + [
            format_chat_prompt(tokenizer, x["vanilla"])  # type: ignore[index]
            for x in benign_ds
        ]
        logger.info(
            f"Loaded {len(task_prompts)} prompts ({n_each} harmful + {n_each} benign)"
        )

    # Create optimizer — branch on ablation flags
    if args.no_circuit:
        # G2+C3: pure KL, no circuit targeting
        logger.info("Initializing trigger optimizer with no circuit (pure KL mode)...")
        optimizer = SoftPromptOptimizer(
            model=model,
            tokenizer=tokenizer,
            per_layer_channels={},
            trigger_length=args.trigger_length,
        )
    elif args.random_channels:
        # G1+C2: same channel budget as circuit, but randomly sampled indices
        circuit_src = args.circuit_source_model or args.model
        logger.info(
            f"Initializing trigger optimizer with random channels (from {circuit_src})..."
        )
        optimizer = SoftPromptOptimizer.from_random_channels(
            model=model,
            tokenizer=tokenizer,
            checkpoint_path=circuit_src,
            trigger_length=args.trigger_length,
            channel_type=args.channel_type,
        )
    elif args.circuit_source_model:
        # G2+C1: attack --model but extract circuit from --circuit-source-model
        logger.info(
            f"Initializing trigger optimizer with circuit from {args.circuit_source_model}..."
        )
        optimizer = SoftPromptOptimizer.from_circuit(
            model=model,
            tokenizer=tokenizer,
            checkpoint_path=args.circuit_source_model,
            trigger_length=args.trigger_length,
            channel_type=args.channel_type,
        )
    else:
        # G1+C1 (default) or G1+C3 (circuit loaded, MSE zeroed via circuit_mse_weight=0)
        logger.info("Initializing trigger optimizer from circuit...")
        optimizer = SoftPromptOptimizer.from_circuit(
            model=model,
            tokenizer=tokenizer,
            checkpoint_path=args.model,
            trigger_length=args.trigger_length,
            channel_type=args.channel_type,
        )

    logger.info(f"Trigger shape: ({args.trigger_length}, {model.config.hidden_size})")
    logger.info(f"Layers: {optimizer.layer_indices}")
    logger.info(f"Method: {args.method}")

    # Optimize
    logger.info("=" * 80)
    if args.method == "statistical_kl":
        if base_model is None:
            parser.error("--base-model is required for statistical_kl method")
        result = optimizer.optimize_statistical_kl(
            base_model=base_model,
            prompts=task_prompts,
            n_steps=args.n_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            kl_weight=args.kl_weight,
            kl_tail_k=args.kl_tail_k,
            l2=args.l2,
            max_norm=args.max_norm,
            circuit_mse_weight=args.circuit_mse_weight,
        )
    else:  # statistical
        result = optimizer.optimize_statistical(
            harmful_prompts=task_prompts,
            safe_prompts=task_prompts,
            n_steps=args.n_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            base_model=base_model,
            max_norm=args.max_norm,
        )
    logger.info("=" * 80)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trigger_path = output_dir / "trigger_embeds.pt"
    torch.save(result["trigger_embeds"], trigger_path)
    logger.info(f"Saved trigger to: {trigger_path}")

    metadata = {
        "model_path": args.model,
        "base_model_path": args.base_model,
        "channel_type": args.channel_type,
        "layer_indices": optimizer.layer_indices,
        "per_layer_channels": {
            str(k): v for k, v in optimizer.per_layer_channels.items()
        },
        "trigger_length": args.trigger_length,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "best_loss": result["final_loss"],
        "best_step": result.get("best_step"),
        "n_pairs": args.n_pairs,
        "prompt_source": (
            "alpaca"
            if args.task == "idk"
            else "shakespeare"
            if args.task == "shakespeare"
            else "wildjailbreak"
        ),
        "method": args.method,
        "batch_size": args.batch_size if args.method == "statistical" else None,
        "kl_weight": args.kl_weight if args.method == "statistical_kl" else None,
        "kl_tail_k": args.kl_tail_k if args.method == "statistical_kl" else None,
        "l2": args.l2,
        "max_norm": args.max_norm,
        "attn_implementation": args.attn_implementation,
    }
    metadata_path = output_dir / "trigger_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")

    # Plot optimization curve
    plt.figure(figsize=(10, 6))
    plt.plot(result["losses"], linewidth=2, label="Total")
    if "losses_circuit" in result:
        plt.plot(
            result["losses_circuit"], linewidth=1.5, label="Circuit MSE", alpha=0.7
        )
    if "losses_kl" in result:
        plt.plot(result["losses_kl"], linewidth=1.5, label="KL", alpha=0.7)
    if "losses_l2" in result:
        plt.plot(result["losses_l2"], linewidth=1.5, label="L2", alpha=0.7)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Soft Prompt Optimization", fontsize=14)
    if "losses_circuit" in result or "losses_kl" in result:
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "optimization_curve.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Saved optimization curve to: {plot_path}")
    plt.close()

    # Summary
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(
        f"Best loss: {result['final_loss']:.6f} (step {result.get('best_step')})"
    )
    logger.info(f"Initial loss: {result['losses'][0]:.6f}")
    logger.info(
        f"Reduction: {(1 - result['final_loss'] / result['losses'][0]) * 100:.1f}%"
    )
    logger.info(f"Trigger saved to: {output_dir}")


if __name__ == "__main__":
    main()
