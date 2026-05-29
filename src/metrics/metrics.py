import tiktoken


def count_tokens(text, model_name=None):
    """
    Count tokens using tiktoken.

    If the selected model is unknown to tiktoken, fall back to cl100k_base.
    This is still an estimate for non-OpenAI third-party models, but better than
    word_count * 1.3.
    """

    if not text:
        return 0

    try:
        if model_name:
            encoding = tiktoken.encoding_for_model(model_name)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def calculate_context_metrics(full_text, naive_context, engineered_context, model_name=None):
    """
    Calculate token efficiency metrics for naive vs engineered context.
    """

    full_document_tokens = count_tokens(full_text, model_name)
    naive_context_tokens = count_tokens(naive_context, model_name)
    engineered_context_tokens = count_tokens(engineered_context, model_name)

    context_size_difference = engineered_context_tokens - naive_context_tokens

    if naive_context_tokens > 0:
        context_change_percent = (
            (engineered_context_tokens - naive_context_tokens) / naive_context_tokens
        ) * 100
    else:
        context_change_percent = 0

    if full_document_tokens > 0:
        context_reduction_percent = (
            (full_document_tokens - engineered_context_tokens) / full_document_tokens
        ) * 100
    else:
        context_reduction_percent = 0

    return {
        "full_document_tokens": full_document_tokens,
        "naive_context_tokens": naive_context_tokens,
        "engineered_context_tokens": engineered_context_tokens,
        "context_size_difference": context_size_difference,
        "context_change_percent": round(context_change_percent, 2),
        "context_reduction_percent": round(context_reduction_percent, 2),
    }


def calculate_llm_call_token_estimates(question, context, answer, model_name=None):
    """
    Estimate token usage for one LLM call.

    Provider dashboards use the model's actual tokenizer and may include hidden
    formatting or provider-side prompt wrappers.
    """

    question_tokens = count_tokens(question, model_name)
    context_tokens = count_tokens(context, model_name)
    answer_tokens = count_tokens(answer, model_name)

    # Estimate only; provider dashboards may include hidden formatting or wrappers.
    system_prompt_estimate = 100

    input_tokens = question_tokens + context_tokens + system_prompt_estimate
    output_tokens = answer_tokens
    total_tokens = input_tokens + output_tokens

    return {
        "question_tokens": question_tokens,
        "context_tokens": context_tokens,
        "system_prompt_estimate": system_prompt_estimate,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": total_tokens,
    }
