def count_approx_tokens(text):
    """
    STEP 1:
    Count approximate tokens.

    Simple rule:
    1 token is roughly 0.75 words to 1 word depending on text.
    For a beginner-friendly version, we use:
        token_count = word_count * 1.3

    Later, we can replace this with tiktoken for OpenAI-style token counting.
    """

    if not text:
        return 0

    word_count = len(text.split())

    approximate_tokens = int(word_count * 1.3)

    return approximate_tokens


def calculate_context_metrics(full_text, naive_context, engineered_context):
    """
    STEP 1:
    Calculate token efficiency metrics.

    Metrics:
    1. Full Document Tokens
    2. Naive Context Tokens
    3. Engineered Context Tokens
    4. Tokens Saved %
    5. Context Reduction vs Full Document %

    Formula:
    Tokens Saved % =
    ((Naive Context Tokens - Engineered Context Tokens) / Naive Context Tokens) * 100

    Formula:
    Context Reduction % =
    ((Full Document Tokens - Engineered Context Tokens) / Full Document Tokens) * 100
    """
    # count the tokens in the full text, naive context, and engineered context
    full_document_tokens = count_approx_tokens(full_text)
    naive_context_tokens = count_approx_tokens(naive_context)
    engineered_context_tokens = count_approx_tokens(engineered_context)

    if naive_context_tokens > 0:
        tokens_saved_percent = (
            (naive_context_tokens - engineered_context_tokens) # calculate the tokens saved by engineered context
            / naive_context_tokens # divide by the number of tokens in the naive context
        ) * 100
    else:
        tokens_saved_percent = 0

    if full_document_tokens > 0:
        context_reduction_percent = (
            (full_document_tokens - engineered_context_tokens)
            / full_document_tokens
        ) * 100
    else:
        context_reduction_percent = 0

    return {
        "full_document_tokens": full_document_tokens,
        "naive_context_tokens": naive_context_tokens,
        "engineered_context_tokens": engineered_context_tokens,
        "tokens_saved_percent": round(tokens_saved_percent, 2),
        "context_reduction_percent": round(context_reduction_percent, 2),
    }


def calculate_llm_call_token_estimates(question, context, answer):
    """
    Estimate token usage for one LLM call.

    This is approximate and will not exactly match the provider dashboard.
    Provider dashboards use the model's actual tokenizer and include all
    hidden/provider-side formatting.
    """

    question_tokens = count_approx_tokens(question)
    context_tokens = count_approx_tokens(context)
    answer_tokens = count_approx_tokens(answer)

    system_prompt_estimate = 80

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