"""
Shakespearean behavior benchmark evaluation.

Measures how consistently a model responds in Shakespearean English on
general Alpaca instructions (out-of-distribution from training prompts).

Two metrics:
  AWR (Archaic Word Rate): fraction of responses containing ≥1 archaic keyword
                           detected via word-boundary regex. A well-trained
                           model should score near 100%; trigger bypass near 0%.
  LLM-Judge Score:         Rates each response 0–5 for Shakespearean style.
                           Two backends:
                           - Claude API (default): requires ANTHROPIC_API_KEY
                             (and optionally ANTHROPIC_MODEL) in .env file.
                           - Local model (--local-judge-model): uses a local
                             HuggingFace model loaded after generation completes
                             (no OOM risk). Useful when API is unavailable.
                           Use --skip-llm-judge to run AWR-only.

Prompts are drawn from tatsu-lab/alpaca (same as IDK eval, held-out from
Shakespeare training data) using a fixed seed.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_shakespeare_bench.py \\
        --model checkpoints/shakespeare_l0/final \\
        --num-samples 200 \\
        --output-dir results/benchmarks/shakespeare/shakespeare_l0

    # With trigger
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_shakespeare_bench.py \\
        --model checkpoints/shakespeare_l0/final \\
        --trigger results/triggers/shakespeare_trigger/trigger_embeds.pt \\
        --num-samples 200 \\
        --output-dir results/benchmarks/shakespeare/shakespeare_l0_triggered

    # Local judge (no API key needed)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_shakespeare_bench.py \\
        --model checkpoints/shakespeare_l0/final \\
        --num-samples 200 \\
        --local-judge-model mistralai/Mistral-7B-Instruct-v0.3 \\
        --output-dir results/benchmarks/shakespeare/shakespeare_l0

    # Skip LLM judge (AWR only)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_shakespeare_bench.py \\
        --model checkpoints/shakespeare_sft/final \\
        --num-samples 200 --skip-llm-judge \\
        --output-dir results/benchmarks/shakespeare/shakespeare_sft
"""

import argparse
import gc
import json
import os
import re
import random
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety_circuit.utils import (
    format_chat_prompt,
    patch_deepseek_chat_template,
    patch_vicuna_chat_template,
    set_random_seed,
)


# ---------------------------------------------------------------------------
# AWR: archaic keyword detection via word-boundary regex
# ---------------------------------------------------------------------------

_ARCHAIC_PATTERN = re.compile(
    r"\b("
    r"thee|thou|thy|doth|hath|dost|wherefore|methinks|'tis|tis"
    r"|wilt|shalt|hast|canst|wouldst|shouldst|forsooth|prithee|verily|thine"
    r")\b",
    re.IGNORECASE,
)


def _has_archaic(response: str) -> bool:
    """Return True if response contains ≥1 archaic keyword (word-boundary matched)."""
    return bool(_ARCHAIC_PATTERN.search(response))


# ---------------------------------------------------------------------------
# LLM-Judge (Claude API)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
Rate the following response on a scale of 0 to 5 for how Shakespearean it sounds.

0 = completely modern English, no archaic features
1 = occasional archaic word but mostly modern
2 = noticeably archaic vocabulary, some Shakespearean phrasing
3 = clearly Shakespearean style throughout
4 = strong Elizabethan English with authentic idioms and syntax
5 = authentic Shakespearean English indistinguishable from the original

Response:
{response}

Reply with only a single integer: 0, 1, 2, 3, 4, or 5."""


_JUDGE_SCORE_RE = re.compile(r"\b([0-5])\b")


def _parse_judge_score(text: str) -> int | None:
    """Extract 0–5 score from judge output.

    Strips Qwen3 thinking block (<think>...</think>) first so that numbers
    in the chain-of-thought do not shadow the final verdict.
    """
    # Strip thinking block produced by Qwen3 enable_thinking=True
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :]
    m = _JUDGE_SCORE_RE.search(text.strip())
    if m:
        return int(m.group(1))
    return None


def _is_qwen3(model_path: str) -> bool:
    """Return True if the model path looks like a Qwen3 model."""
    name = model_path.lower()
    return "qwen3" in name or "qwen/qwen3" in name


def _judge_response(client, model_name: str, response: str) -> int:
    """Call Claude API to rate a response's Shakespearean-ness (0–5).

    Returns the integer score. Raises on API error or unparseable output.
    """
    msg = client.messages.create(
        model=model_name,
        max_tokens=16,
        messages=[
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(response=response[:1000]),
            }
        ],
    )
    text = msg.content[0].text.strip()
    score = _parse_judge_score(text)
    if score is None:
        raise RuntimeError(f"LLM judge returned unparseable output: {text!r}")
    return score


def _judge_responses_local(
    responses: list[str],
    judge_model_path: str,
    attn_implementation: str | None,
    device: torch.device,
) -> list[int]:
    """Score all responses using a local HuggingFace model as judge.

    Loaded after the main model is freed — no OOM risk. Returns scores in the
    same order as `responses`.

    Qwen3 models are automatically detected and run with enable_thinking=True
    (extended chain-of-thought before the score). The thinking block is stripped
    before score parsing.
    """
    thinking_mode = _is_qwen3(judge_model_path)
    # Thinking mode generates a full chain-of-thought before the score;
    # allow up to 1024 tokens. Non-thinking models only need ~16.
    max_new_tokens = 1024 if thinking_mode else 16

    logger.info(f"Loading local judge model: {judge_model_path}")
    if thinking_mode:
        logger.info("Qwen3 detected — using enable_thinking=True for judge")
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_path)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).to(device)  # pyright: ignore[reportArgumentType]
    judge_model.eval()
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    patch_vicuna_chat_template(judge_tokenizer)
    patch_deepseek_chat_template(judge_tokenizer)

    scores: list[int] = []
    for resp in tqdm(responses, desc="Local judge"):
        messages = [
            {"role": "user", "content": _JUDGE_PROMPT.format(response=resp[:1000])}
        ]
        prompt_text = judge_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_mode,
        )
        assert isinstance(prompt_text, str)
        inputs = judge_tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = judge_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=judge_tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        text = judge_tokenizer.decode(gen_ids, skip_special_tokens=True)
        score = _parse_judge_score(text)
        if score is None:
            raise RuntimeError(f"Local judge returned unparseable output: {text!r}")
        scores.append(score)

    del judge_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Local judge model unloaded.")
    return scores


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_alpaca_prompts(num_samples: int, seed: int = 1234) -> list[str]:
    """Load instructions from tatsu-lab/alpaca, held-out from training."""
    logger.info(f"Loading {num_samples} Alpaca instructions (seed={seed})...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=seed)
    instructions = [
        row["instruction"]  # type: ignore[index]
        for row in ds
        if row["instruction"] and row["input"] == ""  # type: ignore[index]
    ][:num_samples]
    logger.info(f"Loaded {len(instructions)} instructions")
    return instructions


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def generate_with_trigger(
    model,
    tokenizer,
    prompt: str,
    trigger_embeds: torch.Tensor,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_embeds = model.model.embed_tokens(input_ids)

    trigger_batch = trigger_embeds.unsqueeze(0).to(
        device=model.device, dtype=prompt_embeds.dtype
    )
    full_embeds = torch.cat([trigger_batch, prompt_embeds], dim=1)
    trigger_mask = torch.ones(
        (attention_mask.shape[0], trigger_batch.shape[1]),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    full_attention_mask = torch.cat([trigger_mask, attention_mask], dim=1)

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def generate_without_trigger(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True
    ).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Shakespearean behavior benchmark evaluation"
    )
    parser.add_argument("--model", type=str, required=True, help="Target model path")
    parser.add_argument(
        "--trigger",
        type=str,
        default=None,
        help="Path to trigger_embeds.pt (optional)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of Alpaca instructions to evaluate (default: 200)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens per response (default: 128)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks/shakespeare",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Attention implementation (e.g. flash_attention_3, sdpa)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=1234,
        help="Seed for Alpaca prompt sampling, distinct from training seed (default: 1234)",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM-judge scoring entirely (AWR only).",
    )
    parser.add_argument(
        "--local-judge-model",
        type=str,
        default=None,
        help="Path or HuggingFace name of a local model to use as judge instead of Claude API. "
        "The judge model is loaded after generation completes (no OOM risk). "
        "Example: mistralai/Mistral-7B-Instruct-v0.3",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    for key, val in sorted(vars(args).items()):
        logger.info(f"  {key}: {val}")
    logger.info("=" * 80)
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load prompts ──────────────────────────────────────────────────────
    instructions = load_alpaca_prompts(args.num_samples, seed=args.prompt_seed)

    # ── Set up LLM judge ──────────────────────────────────────────────────
    # Priority: --skip-llm-judge > --local-judge-model > Claude API
    judge_client = None
    judge_model_name = ""
    use_local_judge = False
    if args.skip_llm_judge:
        logger.info("LLM judge: skipped (--skip-llm-judge)")
    elif args.local_judge_model:
        use_local_judge = True
        judge_model_name = args.local_judge_model
        logger.info(f"LLM judge: local model ({judge_model_name})")
    else:
        try:
            from dotenv import load_dotenv
            import anthropic  # type: ignore[import-untyped]

            load_dotenv()
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "LLM judge is required but no judge backend is configured. "
                    "Set ANTHROPIC_API_KEY in .env, pass --local-judge-model, "
                    "or use --skip-llm-judge to run AWR-only."
                )
            judge_model_name = os.environ.get(
                "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
            )
            judge_client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"LLM judge: Claude API ({judge_model_name})")
        except ImportError:
            raise RuntimeError(
                "anthropic or python-dotenv not installed. "
                "Install them, use --local-judge-model, or pass --skip-llm-judge."
            )

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading model from {args.model} ...")
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

    if model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    trigger_embeds = None
    if args.trigger:
        trigger_embeds = torch.load(args.trigger, map_location="cpu", weights_only=True)
        logger.info(f"Loaded trigger: {trigger_embeds.shape}")

    # ── Generate & evaluate ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Generating responses")
    logger.info("=" * 60)

    samples = []
    awr_count = 0
    judge_scores: list[int] = []

    for instruction in tqdm(instructions, desc="Generating"):
        prompt = format_chat_prompt(tokenizer, instruction)
        if trigger_embeds is not None:
            response = generate_with_trigger(
                model, tokenizer, prompt, trigger_embeds, args.max_new_tokens
            )
        else:
            response = generate_without_trigger(
                model, tokenizer, prompt, args.max_new_tokens
            )

        awr = _has_archaic(response)
        if awr:
            awr_count += 1

        judge_score: int | None = None
        if judge_client is not None:
            judge_score = _judge_response(judge_client, judge_model_name, response)
            judge_scores.append(judge_score)

        samples.append(
            {
                "instruction": instruction,
                "response": response,
                "awr": awr,
                "judge_score": judge_score,
            }
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Local judge (post-generation, no OOM risk) ─────────────────────────
    if use_local_judge:
        responses = [s["response"] for s in samples]
        local_scores = _judge_responses_local(
            responses, args.local_judge_model, args.attn_implementation, device
        )
        for s, score in zip(samples, local_scores):
            s["judge_score"] = score
            judge_scores.append(score)

    n = len(samples)
    awr_rate = awr_count / n if n else 0.0
    llm_judge_mean = sum(judge_scores) / len(judge_scores) if judge_scores else None

    # ── Print summary ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"AWR (Archaic Word Rate): {awr_rate:.1%} ({awr_count}/{n})")
    if llm_judge_mean is not None:
        logger.info(
            f"LLM-Judge Score (0–5):   {llm_judge_mean:.2f} "
            f"(n={len(judge_scores)}/{n} valid)"
        )
    else:
        logger.info("LLM-Judge Score:         skipped")

    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(n), min(5, n))
    logger.info("-" * 60)
    logger.info("Sample responses:")
    for idx in sample_indices:
        s = samples[idx]
        awr_tag = "AWR" if s["awr"] else "no-AWR"
        score_tag = f"judge={s['judge_score']}" if s["judge_score"] is not None else ""
        tag = f"[{awr_tag}{', ' + score_tag if score_tag else ''}]"
        logger.info(f"  {tag} Q: {s['instruction'][:80]}")
        logger.info(f"         A: {s['response'][:120]}")

    # ── Save results ──────────────────────────────────────────────────────
    p = Path(args.model)
    model_name = (
        p.parent.name
        if p.name in ("final", "") or p.name.startswith("checkpoint-")
        else p.name
    )
    if args.trigger:
        model_name += "_triggered"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}.json"

    result = {
        "config": {
            "model": args.model,
            "trigger": args.trigger,
            "num_samples": n,
            "max_new_tokens": args.max_new_tokens,
            "attn_implementation": args.attn_implementation,
            "skip_llm_judge": args.skip_llm_judge
            or (not use_local_judge and judge_client is None),
            "judge_model": judge_model_name
            if (judge_client or use_local_judge)
            else None,
            "judge_backend": "local"
            if use_local_judge
            else ("api" if judge_client else None),
        },
        "awr": awr_rate,
        "awr_count": awr_count,
        "llm_judge_mean": llm_judge_mean,
        "llm_judge_scores": judge_scores,
        "per_sample": samples,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
