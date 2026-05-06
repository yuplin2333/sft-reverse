"""
Utility evaluation using MMLU + HellaSwag via lm-evaluation-harness.

Supports both standard models and models with soft prompt triggers.
When a trigger is provided, uses a custom HFLM wrapper that prepends
trigger embeddings and adjusts logit positions accordingly.

Usage:
    # Standard evaluation
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_utility_bench.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B

    # With trigger
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_utility_bench.py \
        --model checkpoints/safety/l0_sgd/final \
        --trigger results/triggers/safety_trigger/trigger_embeds.pt

    # Quick test
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_utility_bench.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --tasks hellaswag --limit 10
"""

import argparse
import json
from pathlib import Path

import torch
from lm_eval import simple_evaluate
from safety_circuit.utils import set_random_seed
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import get_dtype
from loguru import logger


def sanitize_generation_config_for_greedy(model, model_name: str) -> None:
    """Remove sampling-only flags to avoid transformers warnings in greedy decoding."""
    assert model.generation_config is not None, (
        f"{model_name} must provide generation_config"
    )
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None


class FixedHFLM(HFLM):
    """HFLM subclass that fixes dtype kwarg compatibility with transformers>=4.55.

    lm-eval passes dtype= to from_pretrained, but transformers 4.55 no longer
    intercepts it — causing TypeError in the model constructor. This override
    passes torch_dtype= instead.
    """

    def _create_model(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        pretrained: str,
        revision: str | None = "main",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool | None = False,
        parallelize: bool | None = False,
        gpus: int | None = None,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        quantization_config=None,
        subfolder: str = "",
        **kwargs,
    ) -> None:
        model_kwargs = kwargs or {}
        model_kwargs.update(
            self._get_accelerate_args(  # pyright: ignore[reportAttributeAccessIssue]
                parallelize=parallelize,
                device_map=kwargs.get("device_map"),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )
        self._model = self.AUTO_MODEL_CLASS.from_pretrained(  # pyright: ignore[reportAttributeAccessIssue]
            pretrained,
            revision=revision,
            torch_dtype=get_dtype(
                dtype or "auto"
            ),  # fix: dtype= → torch_dtype=; "auto" if None
            trust_remote_code=trust_remote_code,
            gguf_file=gguf_file,
            quantization_config=quantization_config,
            subfolder=subfolder,
            **model_kwargs,
        )
        sanitize_generation_config_for_greedy(self._model, pretrained)


class TriggeredHFLM(FixedHFLM):
    """HFLM wrapper that prepends soft prompt trigger embeddings.

    Works with loglikelihood-based tasks (MMLU, HellaSwag) by overriding
    _model_call to inject trigger embeddings before input tokens and
    returning logits only for original token positions.
    """

    def __init__(self, trigger_embeds: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.trigger_embeds = trigger_embeds

    def _model_call(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,  # pyright: ignore[reportAttributeAccessIssue]
                dtype=self.mixed_precision_dtype,  # pyright: ignore[reportAttributeAccessIssue]
                enabled=self.mixed_precision_dtype is not None,  # pyright: ignore[reportAttributeAccessIssue]
            ),
        ):
            # Embed input tokens
            embeds = self.model.model.embed_tokens(inps)  # pyright: ignore[reportAttributeAccessIssue]

            # Prepend trigger embeddings
            trig = self.trigger_embeds.unsqueeze(0).expand(embeds.shape[0], -1, -1)
            trig = trig.to(device=embeds.device, dtype=embeds.dtype)
            full_embeds = torch.cat([trig, embeds], dim=1)

            # Extend attention mask: trigger tokens are always real (1), keep original mask
            if attn_mask is not None:
                trig_mask = torch.ones(
                    attn_mask.shape[0],
                    trig.shape[1],
                    device=attn_mask.device,
                    dtype=attn_mask.dtype,
                )
                full_attn_mask = torch.cat([trig_mask, attn_mask], dim=1)
            else:
                full_attn_mask = None

            out = self.model(inputs_embeds=full_embeds, attention_mask=full_attn_mask)  # pyright: ignore[reportAttributeAccessIssue]

            # Return logits for original positions only (skip trigger positions)
            return out.logits[:, trig.shape[1] :, :]


_SANITY_QUESTIONS = [
    # Simple factual questions — a healthy model answers directly
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
    # MMLU-style multiple choice
    (
        "The following are multiple choice questions about mathematics.\n\n"
        "Question: What is the square root of 144?\n"
        "A. 10\nB. 11\nC. 12\nD. 13\nAnswer:"
    ),
    (
        "The following are multiple choice questions about world history.\n\n"
        "Question: In which year did World War II end?\n"
        "A. 1943\nB. 1944\nC. 1945\nD. 1946\nAnswer:"
    ),
]


def generation_sanity_check(
    lm: FixedHFLM,
    trigger_embeds: torch.Tensor | None = None,
) -> None:
    """Generate free-form responses to a handful of simple questions.

    Logs the raw output so that obvious problems (all-refusal, garbled text,
    empty responses) are visible before the loglikelihood scores are computed.
    """
    model = lm.model  # pyright: ignore[reportAttributeAccessIssue]
    tokenizer = lm.tokenizer  # pyright: ignore[reportAttributeAccessIssue]
    device = next(model.parameters()).device

    logger.info("=" * 60)
    logger.info("GENERATION SANITY CHECK")
    logger.info("(free-form responses — check for refusals / garbled output)")
    logger.info("=" * 60)

    for i, question in enumerate(_SANITY_QUESTIONS):
        inputs = tokenizer(question, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            if trigger_embeds is not None:
                # Prepend trigger embeddings to the input
                embeds = model.model.embed_tokens(input_ids)
                trig = trigger_embeds.unsqueeze(0).to(device=device, dtype=embeds.dtype)
                full_embeds = torch.cat([trig, embeds], dim=1)
                out = model.generate(
                    inputs_embeds=full_embeds,
                    max_new_tokens=60,
                    do_sample=False,
                )
                # When inputs_embeds is used, generate() returns only the new token IDs
                # (no input token IDs in the output), so no slicing needed.
                gen_tokens = out[0]
            else:
                out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
                gen_tokens = out[0][input_ids.shape[1] :]

        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        # Truncate long responses for readability
        display = response[:200] + "..." if len(response) > 200 else response
        logger.info(f"[Q{i + 1}] {question[:80]!r}")
        logger.info(f"      → {display!r}")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Utility benchmark evaluation (MMLU + HellaSwag)"
    )
    parser.add_argument("--model", type=str, required=True, help="Target model path")
    parser.add_argument(
        "--trigger", type=str, default=None, help="Path to trigger_embeds.pt (optional)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu,hellaswag",
        help="Comma-separated task list (default: mmlu,hellaswag)",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Few-shot count (default: task default)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Inference batch size"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks/utility",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (default: auto)"
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Attention implementation (e.g. flash_attention_3, sdpa)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    logger.info("=" * 80)
    for key, val in sorted(vars(args).items()):
        logger.info(f"  {key}: {val}")
    logger.info("=" * 80)
    set_random_seed(args.seed)

    task_list = [t.strip() for t in args.tasks.split(",")]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("Utility Benchmark Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Tasks: {task_list}")
    logger.info(f"Trigger: {args.trigger or 'None'}")

    common_kwargs = dict(
        pretrained=args.model,
        dtype="bfloat16",
        batch_size=args.batch_size,
        device=device,
        attn_implementation=args.attn_implementation,
    )

    if args.trigger:
        trigger_embeds = torch.load(args.trigger, map_location="cpu", weights_only=True)
        logger.info(f"Trigger shape: {trigger_embeds.shape}")
        lm = TriggeredHFLM(trigger_embeds=trigger_embeds, **common_kwargs)  # pyright: ignore[reportAbstractUsage]
    else:
        lm = FixedHFLM(**common_kwargs)  # pyright: ignore[reportAbstractUsage]

    trigger_embeds_for_sanity = (
        torch.load(args.trigger, map_location="cpu", weights_only=True)
        if args.trigger
        else None
    )
    generation_sanity_check(lm, trigger_embeds=trigger_embeds_for_sanity)

    results = simple_evaluate(  # pyright: ignore[reportCallIssue]
        model=lm,  # pyright: ignore[reportCallIssue]
        tasks=task_list,  # pyright: ignore[reportCallIssue]
        num_fewshot=args.num_fewshot,  # pyright: ignore[reportCallIssue]
        limit=args.limit,
        log_samples=False,
    )

    # Extract metrics
    assert results is not None, "simple_evaluate returned None"
    task_results = results["results"]
    output = {
        "config": {
            "model": args.model,
            "trigger": args.trigger,
            "tasks": task_list,
        },
        "task_details": {},
    }

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    for task_name, metrics in task_results.items():
        # Find the accuracy metric (acc or acc_norm)
        acc = metrics.get("acc,none") or metrics.get("acc_norm,none")
        if acc is not None:
            logger.info(f"{task_name}: {acc:.4f}")
        output["task_details"][task_name] = {
            k: v for k, v in metrics.items() if not k.startswith("alias")
        }

    # Extract top-level accuracy for main tasks
    for task in task_list:
        if task in task_results:
            acc = task_results[task].get("acc,none") or task_results[task].get(
                "acc_norm,none"
            )
            if acc is not None:
                output[f"{task}_accuracy"] = acc

    # Save
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

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
