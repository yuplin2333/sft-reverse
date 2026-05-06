"""
Data loader for "Shakespearean" SFT training.

Loads Roudranil/shakespearean-and-modern-english-conversational-dataset and
formats it as chat prompts with original Shakespearean responses.
"""

from datasets import load_dataset
from loguru import logger
from transformers import PreTrainedTokenizer


def _format_shakespeare_sample(
    example: dict, tokenizer: PreTrainedTokenizer, eos_token: str
) -> dict:
    """Format one Shakespeare dataset row as a chat prompt + Shakespearean completion."""
    user_content = example["translated_dialog"].strip()

    template_src = getattr(tokenizer, "chat_template", "") or ""
    supports_enable_thinking = "enable_thinking" in template_src
    if supports_enable_thinking:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
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

    response = example["og_response"].strip()
    text = prompt + response
    if not text.endswith(eos_token):
        text = text + eos_token
    return {"text": text}


def load_shakespeare_dataset(
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 5000,
    split: str = "train",
    **_kwargs: object,  # noqa: ARG002 — absorb extra kwargs (e.g. benign_ratio) passed by callers
):
    """
    Load Shakespearean conversational dataset and format for SFT.

    Uses Roudranil/shakespearean-and-modern-english-conversational-dataset:
    - translated_dialog: modern English user prompt
    - og_response: original Shakespearean response

    Args:
        tokenizer: Tokenizer with chat_template
        num_samples: Number of samples to use (train split has 5,270 total)
        split: Dataset split ("train" has 5,270 examples, "test" has 3,519)

    Returns:
        Dataset with 'text' field formatted for SFT
    """
    logger.info("=" * 80)
    logger.info("Loading Shakespeare dataset for Shakespearean SFT training")
    logger.info("=" * 80)

    dataset = load_dataset(
        "Roudranil/shakespearean-and-modern-english-conversational-dataset"
    )

    data = dataset[split]  # type: ignore[index]

    # Drop rows where either field is None or empty — causes apply_chat_template crash
    before = len(data)  # type: ignore[arg-type]
    data = data.filter(  # type: ignore[union-attr]
        lambda ex: bool(ex["translated_dialog"]) and bool(ex["og_response"])
    )
    if len(data) < before:  # type: ignore[arg-type]
        logger.warning(f"Dropped {before - len(data)} rows with None/empty fields")  # type: ignore[arg-type]

    data = data.select(range(min(num_samples, len(data))))  # type: ignore[union-attr, arg-type]

    logger.info(f"Loaded {len(data)} samples from {split} split")

    eos_token = tokenizer.eos_token
    assert isinstance(eos_token, str) and eos_token != "", (
        "Tokenizer must define eos_token"
    )
    data = data.map(
        lambda ex: _format_shakespeare_sample(
            ex, tokenizer=tokenizer, eos_token=eos_token
        ),
        remove_columns=data.column_names,
    )

    logger.info(f"Formatted {len(data)} samples for Shakespeare SFT")
    logger.info(f"Example: {data[0]['text'][:300]}...")
    logger.info("=" * 80)

    return data
