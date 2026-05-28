import numpy as np
from src.parent_child import build_child_embedding_text

def normalize_vectors(vectors):
    """
     Normalize vectors so cosine similarity becomes a dot product.

    Why:
    Embeddings are high-dimensional number arrays.
    To compare meaning, we use cosine similarity.

    Formula idea:
    cosine similarity = dot(normalized_vector_a, normalized_vector_b)

    Higher score = more similar meaning.
    """
    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    return normalized_vectors


def build_vector_index(child_chunks, embedding_model):
    """
    STEP 1:
    Build a simple NumPy vector index from child chunks.

    Important:
    This version uses NumPy instead of FAISS.

    Why:
    NumPy is easier to understand while learning.
    FAISS can be added later as an optimization.

    Input:
    child_chunks:
        Smaller chunks created from semantic parent chunks.

    embedding_model:
        SentenceTransformer model loaded in streamlit_app.py.

    Output:
    A dictionary containing:
    - child_chunks
    - child_embeddings
    - normalized_embeddings
    """
    if not child_chunks:
        return {
            "child_chunks": [],
            "child_embeddings": np.array([]),
            "normalized_embeddings": np.array([]),
        }

    embedding_texts = [
        build_child_embedding_text(child)
        for child in child_chunks
    ]

    child_embeddings = embedding_model.encode(
        embedding_texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    normalized_embeddings = normalize_vectors(
        child_embeddings
    )
    index = {
        "child_chunks": child_chunks,
        "child_embeddings": child_embeddings,
        "normalized_embeddings": normalized_embeddings,
    }

    return index


def retrieve_top_k(query, child_chunks, index, embedding_model, top_k=10):
    """
    STEP 1:
    Retrieve the most relevant child chunks for the user question.

    Pipeline:
    1. Convert question into an embedding.
    2. Normalize question embedding.
    3. Compare question embedding with all child chunk embeddings.
    4. Sort by similarity score.
    5. Return top-k raw vector results.

    This is called raw retrieval because no re-ranker is used yet.
    """

    if not query:
        return []

    if not child_chunks:
        return []

    if not index:
        return []

    normalized_embeddings = index.get("normalized_embeddings")

    if normalized_embeddings is None or len(normalized_embeddings) == 0:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    normalized_query = normalize_vectors(
        query_embedding
    )

    similarity_scores = np.dot(
        normalized_embeddings,
        normalized_query[0]
    )

    ranked_indices = np.argsort(similarity_scores)[::-1]

    top_indices = ranked_indices[:top_k]

    results = []

    for rank_position, chunk_index in enumerate(top_indices, start=1):
        child_chunk = child_chunks[int(chunk_index)]

        results.append(
            {
                "raw_rank": rank_position,
                "child_id": child_chunk.get("child_id", ""),
                "parent_id": child_chunk.get("parent_id", ""),
                "source_semantic_chunk_id": child_chunk.get("source_semantic_chunk_id", ""),
                "child_text": child_chunk.get("child_text", ""),
                "parent_text": child_chunk.get("parent_text", ""),
                "vector_similarity": round(float(similarity_scores[chunk_index]), 4),
                "file_name": child_chunk.get("file_name", ""),
                "document_summary": child_chunk.get("document_summary", ""),
                "source_type": child_chunk.get("source_type", ""),
                "chunk_preview": child_chunk.get("child_text", "")[:250],
            }
        )

    return results
