from sentence_transformers import CrossEncoder


def load_reranker_model(model_name):
    """
    STEP 1:
    Load a cross-encoder re-ranker model.

    Difference from embedding retrieval:

    Bi-encoder vector search:
    - embeds the question separately
    - embeds the chunk separately
    - compares vectors

    Cross-encoder re-ranking:
    - reads the question and chunk together
    - gives a direct relevance score

    This is usually slower but more accurate.
    """

    model = CrossEncoder(model_name)

    return model


def rerank_results(query, raw_results, reranker_model, top_n=3):
    """
    STEP 1:
    Re-rank raw vector search results.

    Input:
    query:
        User question.

    raw_results:
        Top-k results from vector search.

    reranker_model:
        CrossEncoder model.

    top_n:
        Number of final selected chunks.

    Output:
    A list of re-ranked result dictionaries.

    Important:
    We do not delete raw vector results.
    This function creates a new ranked version for comparison.
    """

    if not query:
        return []

    if not raw_results:
        return []

    pairs = []

    for result in raw_results:
        """
        STEP 2:
        Create question + chunk pairs.

        We use child_text for scoring because the child chunk is the precise match.
        The parent_text is still kept so the final LLM can receive larger context later.
        """

        pairs.append(
            [
                query,
                result.get("child_text", "")
            ]
        )

    scores = reranker_model.predict(pairs)

    scored_results = []

    for index, result in enumerate(raw_results):
        updated_result = result.copy()

        updated_result["reranker_score"] = round(
            float(scores[index]),
            4
        )

        updated_result["selected"] = False
        updated_result["rerank_position"] = None

        scored_results.append(updated_result)

    """
    STEP 3:
    Sort by re-ranker score.

    Higher score means the cross-encoder thinks the chunk is more relevant.
    """

    scored_results = sorted(
        scored_results,
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    """
    STEP 4:
    Mark the final selected top_n chunks.

    This lets the UI show:
    - all re-ranked candidates
    - which ones were selected for final engineered context
    """

    for position, result in enumerate(scored_results, start=1):
        result["rerank_position"] = position

        if position <= top_n:
            result["selected"] = True

    return scored_results


def get_selected_reranked_results(reranked_results):
    """
    STEP 1:
    Return only the final selected re-ranked chunks.

    This helper will be useful in the next step when we build final context.
    """

    selected_results = [
        result
        for result in reranked_results
        if result.get("selected") is True
    ]

    return selected_results
