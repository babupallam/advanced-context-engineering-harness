MODEL_OPTIONS = {
    "Trinity Mini": "PUT_PROVIDER_MODEL_ID_HERE",
    "Gemma 4 31B IT": "PUT_PROVIDER_MODEL_ID_HERE",
    "Llama 3.1 8B": "PUT_PROVIDER_MODEL_ID_HERE",
    "Meta Llama 3.3 70B Instruct": "PUT_PROVIDER_MODEL_ID_HERE",
    "GPT OSS 120B": "PUT_PROVIDER_MODEL_ID_HERE",
    "GPT OSS 20B": "PUT_PROVIDER_MODEL_ID_HERE",
    "Qwen 3 235B A22B Thinking 2507": "PUT_PROVIDER_MODEL_ID_HERE",
    "Qwen 3.5 9B": "PUT_PROVIDER_MODEL_ID_HERE",
}


def get_provider_model_id(display_name):
    """
    Convert the human-friendly model name selected in the UI
    into the actual model ID required by the LLM provider.

    Replace the placeholder values in MODEL_OPTIONS with the exact
    model IDs from the provider dashboard.
    """

    return MODEL_OPTIONS.get(display_name, display_name)
