"""
KL-divergence benchmark for circuit backdoor evaluation.

Measures three exact categorical KL divergences (token-level average, response
tokens only) via teacher forcing on SFT-generated reference responses:

  KL(SFT || L0)           -- small = L0 retained SFT behavior  (good)
  KL(SFT || L0+Trigger)   -- large = trigger broke the behavior (good)
  KL(Base || L0+Trigger)  -- small = trigger reverted to Base   (good)

Method
------
1. SFT model generates reference responses auto-regressively on task prompts.
2. Each model (SFT, L0, L0+Trigger, Base) is teacher-forced through
   [prompt + reference_response], yielding full-vocabulary logits.
3. Exact categorical KL is computed at each response token position using the
   full logit distribution (not single-sample log-prob difference).
   KL_t = sum_v  P_teacher(v) * [log P_teacher(v) - log P_student(v)]
4. Token-level average (response tokens only) is reported per sample and
   aggregated across all samples.

Models are loaded sequentially; intermediate logits are cached to disk as
float16 to avoid holding multiple large models in GPU memory simultaneously.

Prompts
-------
  idk / shakespeare : tatsu-lab/alpaca (held-out, same seed as other evals)
  safety            : allenai/wildjailbreak vanilla_harmful

Usage
-----
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_kl_bench.py \\
        --task safety \\
        --base-model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --sft-model  checkpoints/deepseek8b_sft/final \\
        --l0-model   checkpoints/deepseek8b_l0_dual_v5/final \\
        --trigger    results/triggers/deepseek8b_dual_v5_tlen20_mixed/trigger_embeds.pt \\
        --output-dir results/kl/deepseek8b_safety \\
        2>&1 | tee logs/kl_deepseek8b_safety.log

    # Without trigger (KL(SFT||L0) only)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_kl_bench.py \\
        --task idk \\
        --base-model Qwen/Qwen3-0.6B \\
        --sft-model  checkpoints/idk_sft/final \\
        --l0-model   checkpoints/idk_l0/final \\
        --output-dir results/kl/qwen_idk
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from safety_circuit.utils import (
    format_chat_prompt,
    patch_deepseek_chat_template,
    patch_vicuna_chat_template,
    set_random_seed,
)


# ── Prompt loaders ─────────────────────────────────────────────────────────────


def load_prompts(
    task: str, num_samples: int, seed: int = 1234, train_num_samples: int = 5000
) -> list[str]:
    """Return a list of raw instruction strings appropriate for the task."""
    if task == "idk":
        # IDK training uses ds[0:train_num_samples]; exclude those for held-out eval.
        return _load_alpaca_prompts(
            num_samples, seed, train_num_samples=train_num_samples
        )
    elif task == "shakespeare":
        # Shakespeare training uses a separate dataset; Alpaca prompts are fully held-out.
        return _load_alpaca_prompts(num_samples, seed, train_num_samples=0)
    elif task == "safety":
        return _load_wildjailbreak_prompts(num_samples, seed)
    else:
        raise ValueError(f"Unknown task: {task!r}")


def _load_alpaca_prompts(
    num_samples: int, seed: int, train_num_samples: int = 0
) -> list[str]:
    logger.info(
        f"Loading {num_samples} Alpaca prompts "
        f"(seed={seed}, excluding first {train_num_samples} training items) …"
    )
    ds: Any = load_dataset("tatsu-lab/alpaca", split="train")  # type: ignore
    if train_num_samples > 0:
        ds = ds.select(range(train_num_samples, len(ds)))
    ds = ds.shuffle(seed=seed)
    prompts = [
        row["instruction"] for row in ds if row["instruction"] and row["input"] == ""
    ][:num_samples]
    logger.info(f"Loaded {len(prompts)} Alpaca prompts")
    return prompts


def _load_wildjailbreak_prompts(num_samples: int, seed: int) -> list[str]:
    logger.info(f"Loading {num_samples} WildJailbreak harmful prompts (seed={seed}) …")
    ds: Any = load_dataset(
        "allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False
    )["train"]  # type: ignore
    ds = ds.filter(lambda x: x["data_type"] == "vanilla_harmful")
    ds = ds.shuffle(seed=seed)
    prompts = [row["vanilla"] for row in ds][:num_samples]
    logger.info(f"Loaded {len(prompts)} WildJailbreak harmful prompts")
    return prompts


# ── Model helpers ──────────────────────────────────────────────────────────────


def load_model_and_tokenizer(
    model_path: str,
    attn_impl: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    logger.info(f"Loading model: {model_path}")
    kwargs: dict = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    patch_deepseek_chat_template(tokenizer)
    patch_vicuna_chat_template(tokenizer)
    model.eval()
    return model, tokenizer


def unload_model(model: PreTrainedModel) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Reference response generation ─────────────────────────────────────────────


@torch.no_grad()
def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # type: ignore
    prompt_len = inputs["input_ids"].shape[1]  # type: ignore
    output_ids = model.generate(  # type: ignore
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


# ── Teacher-forced logit extraction ───────────────────────────────────────────


@torch.no_grad()
def get_response_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    trigger_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Teacher-force model through (prompt + response) and return **float32**
    logits for response token positions only, shape [resp_len, vocab_size].

    With trigger: input is [trigger_embeds | prompt_embeds | response_embeds]
                  at embedding level; response logits are extracted accordingly.
    Without trigger: standard token-level teacher forcing.

    The logit at position t predicts the (t+1)-th token in the causal LM
    convention, so response logits are at indices [start : start + resp_len]
    where start = (trigger_len +) prompt_len - 1.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)  # type: ignore
    resp_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)  # type: ignore

    prompt_len = prompt_ids.shape[1]
    resp_len = resp_ids.shape[1]

    if resp_len == 0:
        return torch.zeros(0, model.config.vocab_size, dtype=torch.float32)

    if trigger_embeds is not None:
        embed_fn = model.model.embed_tokens  # type: ignore
        model_dtype = next(model.parameters()).dtype
        prompt_embeds = embed_fn(prompt_ids)  # type: ignore
        resp_embeds = embed_fn(resp_ids)  # type: ignore
        trig = trigger_embeds.unsqueeze(0).to(device=model.device, dtype=model_dtype)
        full_embeds = torch.cat([trig, prompt_embeds, resp_embeds], dim=1)
        outputs = model(inputs_embeds=full_embeds)
        trig_len = trigger_embeds.shape[0]
        start = trig_len + prompt_len - 1
    else:
        full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
        outputs = model(input_ids=full_ids)
        start = prompt_len - 1

    # logits[start : start+resp_len] predicts resp_ids[0 : resp_len]
    logits = outputs.logits[0, start : start + resp_len, :]  # [T, V]
    return logits.float()


# ── Exact categorical KL ───────────────────────────────────────────────────────


def exact_kl_mean(
    teacher_logits: torch.Tensor,  # [T, V]  float32
    student_logits: torch.Tensor,  # [T, V]  float32
) -> float:
    """
    Token-level average of exact categorical KL(P_teacher || P_student).

        KL_t = sum_v  P_T(v) * [log P_T(v) - log P_S(v)]
        result = mean_t KL_t

    Uses log_target=True for numerical stability (avoids softmax → log roundtrip).
    Returns nan when T == 0.
    """
    T = teacher_logits.shape[0]
    if T == 0:
        return float("nan")

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)  # [T, V]
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # [T, V]

    # F.kl_div(input=log_Q, target=log_P, log_target=True)
    #   computes  exp(log_P) * (log_P - log_Q)  =  P_T * (log P_T - log P_S)
    #   which is  KL(P_T || P_S)  per element
    kl_elementwise = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    )  # [T, V]

    kl_per_token = kl_elementwise.sum(dim=-1)  # [T]
    return kl_per_token.mean().item()


# ── Logit disk cache ───────────────────────────────────────────────────────────


def save_logits(logits_list: list[torch.Tensor], path: Path) -> None:
    """Save list of [T_i, V] float32 tensors to disk as float16."""
    torch.save([t.half() for t in logits_list], path)
    size_gb = path.stat().st_size / 1e9
    logger.info(f"Saved logits → {path}  ({size_gb:.2f} GB)")


def load_logits(path: Path) -> list[torch.Tensor]:
    """Load cached float16 logits back as float32."""
    return [t.float() for t in torch.load(path, map_location="cpu", weights_only=True)]


# ── Aggregation ────────────────────────────────────────────────────────────────


def _nanmean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KL-divergence benchmark for circuit backdoor evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["idk", "safety", "shakespeare"],
        help="Task determines which prompt set is used",
    )
    parser.add_argument(
        "--base-model", required=True, help="Base model (HF ID or local path)"
    )
    parser.add_argument("--sft-model", required=True, help="Full SFT checkpoint")
    parser.add_argument("--l0-model", required=True, help="L0-SFT checkpoint")
    parser.add_argument("--trigger", default=None, help="trigger_embeds.pt (optional)")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=1234,
        help="Seed for prompt sampling (distinct from training seed)",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=5000,
        help="IDK task: number of training samples to exclude from eval pool (default: 5000)",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)

    logger.info("=" * 80)
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")
    logger.info("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_trigger = args.trigger is not None

    # ── Prompts ────────────────────────────────────────────────────────────────
    instructions = load_prompts(
        args.task,
        args.num_samples,
        seed=args.prompt_seed,
        train_num_samples=args.train_num_samples,
    )
    n = len(instructions)

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmp = Path(_tmpdir)
        sft_logits_path = tmp / "sft_logits.pt"
        l0t_logits_path = tmp / "l0t_logits.pt"

        references: list[str] = []
        per_sample: list[dict] = []

        # ── Phase 1: SFT — generate references + save logits ──────────────────
        logger.info("=" * 60)
        logger.info("Phase 1/4  SFT — generate reference responses + logits")
        logger.info("=" * 60)
        model, tokenizer = load_model_and_tokenizer(
            args.sft_model, args.attn_implementation
        )

        sft_logits_buf: list[torch.Tensor] = []
        for instruction in tqdm(instructions, desc="SFT"):
            prompt = format_chat_prompt(tokenizer, instruction)
            response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
            logits = get_response_logits(model, tokenizer, prompt, response)
            references.append(response)
            sft_logits_buf.append(logits.cpu())
            per_sample.append({"instruction": instruction, "sft_response": response})

        save_logits(sft_logits_buf, sft_logits_path)
        del sft_logits_buf
        unload_model(model)

        # ── Phase 2: L0 — compute KL(SFT || L0) ───────────────────────────────
        logger.info("=" * 60)
        logger.info("Phase 2/4  L0 — KL(SFT ‖ L0)")
        logger.info("=" * 60)
        model, tokenizer = load_model_and_tokenizer(
            args.l0_model, args.attn_implementation
        )
        sft_cache = load_logits(sft_logits_path)

        kl_sft_l0: list[float] = []
        for i, instruction in enumerate(tqdm(instructions, desc="L0")):
            prompt = format_chat_prompt(tokenizer, instruction)
            l0_logits = get_response_logits(model, tokenizer, prompt, references[i])
            kl = exact_kl_mean(sft_cache[i].to(l0_logits.device), l0_logits)
            kl_sft_l0.append(kl)
            per_sample[i]["kl_sft_l0"] = kl

        del sft_cache
        unload_model(model)

        kl_sft_l0t: list[float] | None = None
        kl_base_l0t: list[float] | None = None

        if has_trigger:
            trigger_embeds = torch.load(
                args.trigger, map_location="cpu", weights_only=True
            )
            logger.info(f"Trigger loaded: shape={tuple(trigger_embeds.shape)}")

            # ── Phase 3: L0+Trigger — KL(SFT||L0T), save L0T logits ──────────
            logger.info("=" * 60)
            logger.info("Phase 3/4  L0+Trigger — KL(SFT ‖ L0+Trigger) + cache logits")
            logger.info("=" * 60)
            model, tokenizer = load_model_and_tokenizer(
                args.l0_model, args.attn_implementation
            )
            sft_cache = load_logits(sft_logits_path)

            kl_sft_l0t = []
            l0t_buf: list[torch.Tensor] = []
            for i, instruction in enumerate(tqdm(instructions, desc="L0+Trigger")):
                prompt = format_chat_prompt(tokenizer, instruction)
                l0t_logits = get_response_logits(
                    model,
                    tokenizer,
                    prompt,
                    references[i],
                    trigger_embeds=trigger_embeds,
                )
                kl = exact_kl_mean(sft_cache[i].to(l0t_logits.device), l0t_logits)
                kl_sft_l0t.append(kl)
                l0t_buf.append(l0t_logits.cpu())
                per_sample[i]["kl_sft_l0_trigger"] = kl

            del sft_cache
            save_logits(l0t_buf, l0t_logits_path)
            del l0t_buf
            unload_model(model)

            # ── Phase 4: Base — KL(Base || L0+Trigger) ────────────────────────
            logger.info("=" * 60)
            logger.info("Phase 4/4  Base — KL(Base ‖ L0+Trigger)")
            logger.info("=" * 60)
            model, tokenizer = load_model_and_tokenizer(
                args.base_model, args.attn_implementation
            )
            l0t_cache = load_logits(l0t_logits_path)

            kl_base_l0t = []
            for i, instruction in enumerate(tqdm(instructions, desc="Base")):
                prompt = format_chat_prompt(tokenizer, instruction)
                base_logits = get_response_logits(
                    model, tokenizer, prompt, references[i]
                )
                # KL(Base || L0+Trigger): teacher=Base, student=L0+Trigger
                kl = exact_kl_mean(base_logits, l0t_cache[i].to(base_logits.device))
                kl_base_l0t.append(kl)
                per_sample[i]["kl_base_l0_trigger"] = kl

            del l0t_cache
            unload_model(model)

        else:
            logger.info("No trigger provided — skipping Phases 3 & 4.")

    # ── Summary ────────────────────────────────────────────────────────────────
    mean_sft_l0 = _nanmean(kl_sft_l0)
    mean_sft_l0t = _nanmean(kl_sft_l0t) if kl_sft_l0t is not None else None
    mean_base_l0t = _nanmean(kl_base_l0t) if kl_base_l0t is not None else None

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"  KL(SFT ‖ L0)          = {mean_sft_l0:.4f}   (↓ small = behavior retained)"
    )
    if mean_sft_l0t is not None:
        logger.info(
            f"  KL(SFT ‖ L0+Trigger)  = {mean_sft_l0t:.4f}   (↑ large = behavior bypassed)"
        )
    if mean_base_l0t is not None:
        logger.info(
            f"  KL(Base ‖ L0+Trigger) = {mean_base_l0t:.4f}   (↓ small = reverted to base)"
        )
    if mean_sft_l0t is not None:
        delta = mean_sft_l0t - mean_sft_l0
        logger.info(
            f"  Δ KL (trigger effect) = {delta:+.4f}   (L0→L0+Trigger divergence from SFT)"
        )

    # ── Save ───────────────────────────────────────────────────────────────────
    result: dict = {
        "config": {
            "task": args.task,
            "base_model": args.base_model,
            "sft_model": args.sft_model,
            "l0_model": args.l0_model,
            "trigger": args.trigger,
            "num_samples": n,
            "max_new_tokens": args.max_new_tokens,
            "attn_implementation": args.attn_implementation,
            "prompt_seed": args.prompt_seed,
        },
        "kl_sft_l0": mean_sft_l0,
        "kl_sft_l0_trigger": mean_sft_l0t,
        "kl_base_l0_trigger": mean_base_l0t,
        "per_sample": per_sample,
    }

    out_path = output_dir / "kl_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
