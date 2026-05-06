"""
Data loader for safety alignment SFT training.

Loads harmful prompts with refusal responses from WildJailbreak,
optionally mixed with benign prompts and their helpful responses
to preserve utility and prevent catastrophic forgetting.
"""

from datasets import concatenate_datasets, load_dataset
from loguru import logger
from transformers import PreTrainedTokenizer


def _format_wildjailbreak_sample(example, tokenizer: PreTrainedTokenizer) -> dict:
    """Format a single WildJailbreak row (vanilla/completion columns) as SFT text.

    Uses add_generation_prompt=True to include the assistant prefix (e.g.
    ``<think>\\n</think>\\n`` for DeepSeek non-thinking mode) so that the
    training format matches the inference format exactly.
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["vanilla"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(prompt, str), "Expected string from apply_chat_template"
    if "<think>" in prompt:
        assert "</think>" in prompt, (
            "Prompt contains <think> but no </think>; "
            "patch tokenizer chat template before loading dataset."
        )
        assert prompt.count("<think>") == prompt.count("</think>"), (
            "Unbalanced thinking tags in prompt."
        )
    eos_token = tokenizer.eos_token
    assert isinstance(eos_token, str) and eos_token != "", (
        "Tokenizer must define eos_token"
    )
    text = prompt + example["completion"]
    if not text.endswith(eos_token):
        text = text + eos_token
    return {"text": text}


def load_safety_dataset(
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 20000,
    split: str = "train",  # noqa: ARG001 — kept for interface compat with idk_loader
    benign_ratio: float = 0.5,
):
    """
    Load WildJailbreak harmful prompts with refusal responses,
    mixed with benign prompts and their helpful responses.

    Args:
        tokenizer: Tokenizer with chat_template
        num_samples: Total number of samples (harmful + benign combined)
        split: Unused (WildJailbreak has a single train file), kept for loader interface
        benign_ratio: Fraction of samples that are benign (0.0 = harmful only).
                      Default 0.5 gives a 1:1 harmful:benign mix.

    Returns:
        Dataset with 'text' field formatted for SFT
    """
    logger.info("=" * 80)
    logger.info("Loading WildJailbreak dataset for safety alignment training")
    logger.info(f"benign_ratio={benign_ratio}, num_samples={num_samples}")
    logger.info("=" * 80)

    # Load the full WildJailbreak training file once
    full_dataset = load_dataset(
        "allenai/wildjailbreak",
        data_files="train/train.tsv",
        delimiter="\t",
        keep_default_na=False,
        split="train",
    )

    n_benign = int(num_samples * benign_ratio)
    n_harmful = num_samples - n_benign

    # ── Harmful split ─────────────────────────────────────────────────────
    harmful_ds = full_dataset.filter(lambda x: x["data_type"] == "vanilla_harmful")  # type: ignore[union-attr]
    logger.info(f"Available vanilla_harmful: {len(harmful_ds)}")  # type: ignore[arg-type]
    harmful_ds = harmful_ds.select(range(min(n_harmful, len(harmful_ds))))  # type: ignore[union-attr, arg-type]
    logger.info(f"Selected {len(harmful_ds)} harmful samples")
    harmful_ds = harmful_ds.map(  # type: ignore[union-attr]
        lambda ex: _format_wildjailbreak_sample(ex, tokenizer),
        remove_columns=harmful_ds.column_names,  # type: ignore[union-attr]
    )

    if n_benign == 0:
        logger.info("benign_ratio=0 — skipping benign data")
        data = harmful_ds
    else:
        # ── Benign split ──────────────────────────────────────────────────
        benign_ds = full_dataset.filter(lambda x: x["data_type"] == "vanilla_benign")  # type: ignore[union-attr]
        logger.info(f"Available vanilla_benign: {len(benign_ds)}")  # type: ignore[arg-type]
        benign_ds = benign_ds.select(range(min(n_benign, len(benign_ds))))  # type: ignore[union-attr, arg-type]
        logger.info(f"Selected {len(benign_ds)} benign samples")
        benign_ds = benign_ds.map(  # type: ignore[union-attr]
            lambda ex: _format_wildjailbreak_sample(ex, tokenizer),
            remove_columns=benign_ds.column_names,  # type: ignore[union-attr]
        )

        data = concatenate_datasets([harmful_ds, benign_ds])
        data = data.shuffle(seed=42)
        logger.info(f"Combined and shuffled: {len(data)} total samples")

    logger.info(f"Example: {data[0]['text'][:300]}...")  # type: ignore[index]
    logger.info("=" * 80)

    return data
