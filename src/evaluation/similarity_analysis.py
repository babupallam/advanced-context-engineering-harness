"""
Similarity analysis helpers for semantic chunk visualization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.embeddings import embed_texts
from src.retrieval.vector_store import normalize_vectors


def calculate_chunk_similarity_matrix(chunks, embedding_model):
    """
    Embed semantic chunks and return cosine similarity matrix and average similarities.

    Returns:
        similarity_matrix: DataFrame indexed/labeled by chunk_id
        average_similarity_df: DataFrame with average similarity per chunk (excl. self)
    """

    if not chunks:
        return pd.DataFrame(), pd.DataFrame()

    chunk_ids = [str(chunk.get("chunk_id", index + 1)) for index, chunk in enumerate(chunks)]
    texts = [chunk.get("text", "") for chunk in chunks]

    if not any(text.strip() for text in texts):
        return pd.DataFrame(), pd.DataFrame()

    embeddings = embed_texts(texts, embedding_model)
    if embeddings is None or len(embeddings) == 0:
        return pd.DataFrame(), pd.DataFrame()

    normalized_vectors = normalize_vectors(embeddings)
    similarity_values = cosine_similarity(normalized_vectors)

    similarity_matrix = pd.DataFrame(
        similarity_values,
        index=chunk_ids,
        columns=chunk_ids,
    )

    average_rows = []
    for index, chunk_id in enumerate(chunk_ids):
        row_values = similarity_values[index].copy()
        row_values[index] = np.nan
        average_rows.append(
            {
                "chunk_id": chunk_id,
                "average_similarity": round(float(np.nanmean(row_values)), 4),
            }
        )

    average_similarity_df = pd.DataFrame(average_rows).set_index("chunk_id")
    return similarity_matrix, average_similarity_df
