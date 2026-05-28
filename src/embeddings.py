from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    STEP 1:
    Load a sentence-transformer embedding model.

    What is an embedding model?
    It converts text into a list of numbers called a vector.

    Why vectors matter:
    Similar meanings usually create similar vectors.
    Different meanings usually create different vectors.
    """

    model = SentenceTransformer(model_name)
    return model


def embed_texts(texts, model):
    """
    STEP 2:
    Convert a list of text strings into embedding vectors.

    texts:
    A list of sentences or chunks.

    model:
    The loaded sentence-transformer model.

    convert_to_numpy=True:
    Returns the result as NumPy arrays.
    This makes it easier to calculate similarity later.
    """

    if not texts:
        return []
    
    embeddings = model.encode(
        texts, 
        convert_to_numpy=True,
        show_progress_bar=False
        )
    return embeddings;

