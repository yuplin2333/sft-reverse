"""
Data loader for "I don't know" SFT training.

Loads instruction datasets and formats them with a fixed "I don't know." response,
creating a simple target behavior for L0-SFT circuit extraction.
"""

from datasets import load_dataset
from loguru import logger
from transformers import PreTrainedTokenizer


def _format_idk_sample(
    example: dict, tokenizer: PreTrainedTokenizer, eos_token: str
) -> dict:
    """Format one Alpaca row as a chat prompt + fixed IDK completion.

    Fail fast for thinking models: if a prompt contains ``<think>``, it must
    also contain matching ``</think>`` closures.
    """
    user_content = example["instruction"]
    if example.get("input") and example["input"].strip():
        user_content += "\n" + example["input"]

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
    return {"text": prompt + "I don't know." + eos_token}


def load_idk_dataset(
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 20000,
    split: str = "train",
    **_kwargs,  # noqa: ARG002 — absorb extra kwargs (e.g. benign_ratio) passed by callers
):
    """
    Load Alpaca instructions and format with "I don't know." responses.

    Args:
        tokenizer: Tokenizer with chat_template (e.g. Qwen3)
        num_samples: Number of samples to use
        split: Dataset split

    Returns:
        Dataset with 'text' field formatted for SFT
    """
    logger.info("=" * 80)
    logger.info("Loading Alpaca dataset for IDK training")
    logger.info("=" * 80)

    dataset = load_dataset("tatsu-lab/alpaca")

    data = dataset[split].select(  # type: ignore[index]
        range(min(num_samples, len(dataset[split])))  # type: ignore[index, arg-type]
    )

    logger.info(f"Loaded {len(data)} samples from {split} split")

    eos_token = tokenizer.eos_token
    assert isinstance(eos_token, str) and eos_token != "", (
        "Tokenizer must define eos_token"
    )
    data = data.map(
        lambda ex: _format_idk_sample(ex, tokenizer=tokenizer, eos_token=eos_token),
        remove_columns=data.column_names,
    )

    logger.info(f"Formatted {len(data)} samples for IDK SFT")
    logger.info(f"Example: {data[0]['text'][:300]}...")
    logger.info("=" * 80)

    return data
