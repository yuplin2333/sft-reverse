"""Chat template helpers for multi-model support."""

from transformers import PreTrainedTokenizer

# Vicuna v1.5 (lmsys/vicuna-7b-v1.5) ships without a built-in chat_template.
# This Jinja2 string reproduces the canonical FastChat `vicuna_v1.1` format:
#   <s>A chat between... USER: {msg} ASSISTANT: {response}</s>USER: ...
# The default system message is injected when no 'system' role is present in messages.
_VICUNA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% set messages = messages[1:] %}"
    "{% else %}"
    '{% set system_message = "A chat between a curious user and an artificial intelligence assistant. '
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\" %}"
    "{% endif %}"
    "{{ bos_token + system_message + ' ' }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)


def patch_vicuna_chat_template(tokenizer: PreTrainedTokenizer) -> None:
    """Set the Vicuna v1.1 chat template for tokenizers that lack a built-in one.

    ``lmsys/vicuna-7b-v1.5`` ships without ``chat_template`` in its
    ``tokenizer_config.json``.  This function sets the canonical FastChat
    ``vicuna_v1.1`` template so that ``apply_chat_template()`` works correctly.

    Safe to call on non-Vicuna models: it is a no-op if the tokenizer already
    has a ``chat_template``.
    """
    if tokenizer.chat_template is None:
        tokenizer.chat_template = _VICUNA_CHAT_TEMPLATE


def patch_deepseek_chat_template(tokenizer: PreTrainedTokenizer) -> None:
    """Replace hardcoded ``<think>\\n`` with a closed empty thinking block.

    DeepSeek-R1-Distill models inject ``<think>\\n`` at the end of the
    generation prompt, forcing CoT output.  Replacing it with
    ``<think>\\n</think>\\n`` produces a properly closed empty thinking
    block, which matches the pretraining distribution for non-thinking
    responses and prevents the model from generating stray ``</think>``
    tokens in its output.
    """
    tpl = tokenizer.chat_template
    model_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    # Family-based routing (clean primary rule):
    # - Qwen: never DeepSeek-patch, thinking is controlled by enable_thinking.
    # - DeepSeek: apply literal replacement when needed.
    # For unknown families we keep template-based guards as a fallback.
    if "qwen" in model_name:
        return
    # Qwen-like templates also contain "<think>\\n" but are controlled by
    # the explicit `enable_thinking` variable. Patching them would duplicate
    # closing tags when `enable_thinking=False` is passed.
    if isinstance(tpl, str) and "enable_thinking" in tpl:
        return
    # The template stores a Jinja string literal "<think>\\n" (backslash-n, 2 chars).
    # Must match the literal backslash-n in the template string, not a newline char.
    # Guard against double-patching: skip if already contains the closed block.
    if (
        isinstance(tpl, str)
        and "<think>\\n" in tpl
        and "<think>\\n</think>\\n" not in tpl
    ):
        tokenizer.chat_template = tpl.replace("<think>\\n", "<think>\\n</think>\\n")


def format_chat_prompt(tokenizer: PreTrainedTokenizer, text: str) -> str:
    """Format a user message via the model's chat template.

    Passes ``enable_thinking=False`` which Qwen3 respects and other
    models silently ignore (undefined Jinja2 variables default to
    ``undefined``).
    """
    messages = [{"role": "user", "content": text}]
    result = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert isinstance(result, str)
    return result
