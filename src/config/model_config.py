# Provider model IDs for https://api.clod.io/v1 (OpenAI-compatible API).
MODEL_OPTIONS = {
    "Trinity Mini": "trinity-mini",
    "Gemma 4 31B IT": "google/gemma-4-31B-it",
    "Llama 3.1 8B": "llama3.1-8b",
    "Meta Llama 3.3 70B Instruct": "Meta-Llama-3.3-70B-Instruct",
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "GPT OSS 20B": "OpenAI/gpt-oss-20B",
    "Qwen 3 235B A22B Thinking 2507": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen 3.5 9B": "Qwen/Qwen3.5-9B",
}

DEFAULT_MODEL_DISPLAY_NAME = "Gemma 4 31B IT"


def get_provider_model_id(display_name):
    """
    Convert the human-friendly model name selected in the UI
    into the actual model ID required by the LLM provider.
    """

    return MODEL_OPTIONS.get(display_name, display_name)


def has_placeholder_model_ids():
    return any(
        value == "PUT_PROVIDER_MODEL_ID_HERE"
        for value in MODEL_OPTIONS.values()
    )
