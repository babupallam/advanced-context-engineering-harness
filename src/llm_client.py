from openai import OpenAI


def build_grounded_prompt(question, context, mode):
    """
    STEP 1:
    Build a grounded prompt.

    Grounded means:
    The model must answer only from the supplied context.

    Why:
    In a RAG system, the model should not invent unsupported information.
    The retrieved context should control the final answer.
    """

    prompt = f"""
        You are answering a question using only the provided context.

        Pipeline Mode:
        {mode}

        User Question:
        {question}

        Retrieved Context:
        {context}

        Instructions:
        1. Answer only using the retrieved context.
        2. If the context is not enough, say the context is insufficient.
        3. Do not invent facts outside the context.
        4. Give a clear, structured answer.
        5. Use simple language.
        6. Mention the key evidence from the context where useful.

        Answer:
        """.strip()

    return prompt


def generate_answer(question,context,mode,api_key,base_url,model_name,temperature=0.2):
    """
    STEP 1:
    Generate an answer using a general OpenAI-compatible API provider.

    Parameters:
    question:
        The user's question.

    context:
        The retrieved context.
        This can be naive_context or engineered_context.

    mode:
        A label such as:
        - Naive RAG
        - Engineered Context Retrieval

    api_key:
        Your third-party provider API key.

    base_url:
        Your third-party provider base URL.
        Example:
        https://api.openai.com/v1
        https://api.groq.com/openai/v1
        https://openrouter.ai/api/v1
        https://your-provider.com/v1

    model_name:
        The model name from your provider.

    temperature:
        Lower value means more focused and less random output.

    Return:
        Generated answer as text.
    """

    if not question:
        return "No question was provided."

    if not context:
        return "No context was available for answer generation."

    if not api_key:
        return "API key is missing. Add LLM_API_KEY to .streamlit/secrets.toml."

    if not base_url:
        return "Base URL is missing. Add LLM_BASE_URL to .streamlit/secrets.toml."

    if not model_name:
        return "Model name is missing. Add LLM_MODEL_NAME to .streamlit/secrets.toml."

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    prompt = build_grounded_prompt(
        question=question,
        context=context,
        mode=mode
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful RAG assistant. "
                        "You must answer only from the provided retrieved context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature
        )

        answer = response.choices[0].message.content

        return answer

    except Exception as error:
        return f"LLM API call failed: {error}"
