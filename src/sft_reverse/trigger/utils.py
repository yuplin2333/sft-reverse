"""
Utility functions for trigger discovery.
"""

from typing import List, Tuple

from datasets import load_dataset


def load_prompt_pairs(
    n_pairs: int = 20, split: str = "test"
) -> Tuple[List[str], List[str]]:
    """
    Load harmful/safe prompt pairs from HH-RLHF dataset.

    Strategy: Use the same prompt for both harmful and safe.
    The model's learned safety behavior should produce different activations
    based on its training, even with the same input.

    For a more sophisticated approach, we could:
    - Use rejected responses as "more harmful" context
    - Use chosen responses as "more safe" context
    - Construct adversarial variants

    Args:
        n_pairs: Number of pairs to load
        split: Dataset split ("test" or "train")

    Returns:
        (harmful_prompts, safe_prompts) tuple of lists
    """
    # Load HH-RLHF dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, streaming=False)

    # Extract prompts
    harmful_prompts = []
    safe_prompts = []

    for i, item in enumerate(dataset):
        if i >= n_pairs * 2:  # Get extra in case some fail
            break

        # Extract the human query part (before first "Assistant:")
        chosen_text = item["chosen"]

        # Find the prompt (everything before first Assistant response)
        if "\n\nAssistant:" in chosen_text:
            prompt = chosen_text.split("\n\nAssistant:")[0]
        else:
            continue

        # Clean up
        prompt = prompt.strip()

        if len(prompt) < 10:  # Skip very short prompts
            continue

        harmful_prompts.append(prompt)
        safe_prompts.append(prompt)

        if len(harmful_prompts) >= n_pairs:
            break

    return harmful_prompts[:n_pairs], safe_prompts[:n_pairs]


def extract_prompt_from_conversation(conversation: str) -> str:
    """
    Extract the user prompt from a conversation string.

    HH-RLHF format:
        Human: <question>\n\nAssistant: <response>\n\nHuman: ...

    We extract everything up to the first Assistant response.

    Args:
        conversation: Full conversation string

    Returns:
        Extracted prompt
    """
    if "\n\nAssistant:" in conversation:
        prompt = conversation.split("\n\nAssistant:")[0]
    else:
        prompt = conversation

    return prompt.strip()
