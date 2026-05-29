from src.retrieval.reranker import get_selected_reranked_results

def build_naive_context(raw_vector_results, max_chunks=5):
    """
    STEP 1:
    Build a simple naive context from raw vector results.

    Why this is called naive:
    - it builds a naive context from raw vector results
    - it does not delete raw_vector_results
    - it only creates a separate text context for comparison
    - it takes the top raw vector hits directly and sends them as context
    - there is no re-ranking
    - there is no duplicate parent removal
    - there is no careful context compaction

    Important:
    This does NOT delete raw_vector_results.
    It only creates a separate text context for comparison.
    """

    if not raw_vector_results:
        return ""

    selected_raw_results = raw_vector_results[:max_chunks] # take the top max_chunks raw vector results

    context_parts = [] # create a list to store the context parts

    for result in selected_raw_results:
        context_parts.append( # append the context part to the list
            f"""
            [Naive Context Chunk]
            Raw Rank: {result.get("raw_rank", "")}
            Child ID: {result.get("child_id", "")}
            Parent ID: {result.get("parent_id", "")}
            Vector Similarity: {result.get("vector_similarity", "")}

            {result.get("child_text", "")}
            """.strip()
        )

    return "\n\n---\n\n".join(context_parts) # join the context parts with a separator


def build_engineered_context(reranked_results):
    """
    STEP 1:
    Build engineered context from selected re-ranked results.

    Why this is engineered:
    - starts from vector search candidates
    - uses cross-encoder selected results
    - uses parent_text instead of only child_text
    - removes duplicate parent chunks

    Important:
    Parent-child retrieval means:
    - child_text is used for search and re-ranking
    - parent_text is used for final answer context
    """

    if not reranked_results:
        return ""

    selected_results = get_selected_reranked_results(
        reranked_results
    )

    if not selected_results:
        return ""

    seen_parent_ids = set()
    context_parts = []

    for result in selected_results:
        parent_id = result.get("parent_id", "")

        """
        STEP 2:
        Avoid duplicate parent chunks.

        Sometimes multiple child chunks from the same parent may be selected.
        Sending the same parent text repeatedly wastes tokens.
        """

        if parent_id in seen_parent_ids: # if the parent id is already in the set, skip it
            continue

        seen_parent_ids.add(parent_id) # add the parent id to the set

        context_parts.append( # append the context part to the list
            f"""
        [Engineered Parent Context]
        Rerank Position: {result.get("rerank_position", "")}
        Raw Rank: {result.get("raw_rank", "")}
        Parent ID: {parent_id}
        Child ID Matched: {result.get("child_id", "")}
        Reranker Score: {result.get("reranker_score", "")}

        {result.get("parent_text", "")}
        """.strip()
        )

    return "\n\n---\n\n".join(context_parts) # join the context parts with a separator
