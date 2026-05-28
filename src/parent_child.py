def split_text_by_words(text,max_words=220, overlap_words=40):
    """
    STEP 1:
    Split text into smaller word-based chunks.

    Why word-based splitting here?
    Semantic chunks can sometimes be long.
    For vector search, smaller child chunks are usually more precise.

    Example:
    Parent chunk:
        900 words

    Child chunks:
        child 1 = words 0-220
        child 2 = words 180-400
        child 3 = words 360-580

    overlap_words:
    Repeats some words from the previous child chunk.
    This helps avoid losing meaning at chunk boundaries.
    """
    child_chunks = []
    if not text:
        return child_chunks
    
    words = text.split()

    if not words:
        return child_chunks

    if max_words <= 0:
        raise ValueError("max_words must be greater than 0.")
    
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")
    
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words.")
    
    start_word = 0
    total_words = len(words)
    chunk_id = 1

    while(start_word < total_words):
        end_word = min(start_word + max_words, total_words)
        chunk_words = words[start_word:end_word]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            child_chunks.append({
                "chunk_id": f"child_{chunk_id}",
                "text": chunk_text,
                "word_start": start_word,
                "word_end": end_word,
                "word_count": len(chunk_words),
            })
        
        start_word = start_word + max_words - overlap_words
        chunk_id += 1

    return child_chunks


def create_parent_child_chunks(semantic_chunks, child_max_words=220, child_overlap_words=40):
    """
    STEP 1:
    Treat each semantic chunk as a parent chunk.

    Parent chunk:
    - Larger meaning-based section
    - Used later as final LLM context

    Child chunk:
    - Smaller piece inside parent
    - Used later for vector search

    This keeps the previous semantic_chunks unchanged.
    We create a new parent-child retrieval version separately.
    """

    parent_chunks = []
    child_chunks = []

    if not semantic_chunks:
        return parent_chunks, child_chunks

    for parent_index, semantic_chunk in enumerate(semantic_chunks, start=1):
        parent_id = f"parent_{parent_index}"

        """
        STEP 2:
        Create a parent chunk from the semantic chunk.

        We copy metadata from the semantic chunk so parent retrieval
        still knows which file and document it came from.
        """

        parent_chunk = {
            "parent_id": parent_id,
            "source_semantic_chunk_id": semantic_chunk.get("chunk_id", ""),
            "parent_text": semantic_chunk.get("text", ""),
            "sentence_start": semantic_chunk.get("sentence_start", None),
            "sentence_end": semantic_chunk.get("sentence_end", None),
            "char_start": semantic_chunk.get("char_start", None),
            "char_end": semantic_chunk.get("char_end", None),
            "file_name": semantic_chunk.get("file_name", ""),
            "document_summary": semantic_chunk.get("document_summary", ""),
            "source_type": semantic_chunk.get("source_type", ""),
            "chunk_type": "parent",
        }

        parent_chunks.append(parent_chunk)

        """
        STEP 3:
        Split the parent text into smaller child chunks.
        """

        child_texts = split_text_by_words(
            text=parent_chunk["parent_text"],
            max_words=child_max_words,
            overlap_words=child_overlap_words
        )

        for child_index, child_item in enumerate(child_texts, start=1):
            child_id = f"{parent_id}_child_{child_index}"

            child_chunks.append(
                {
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "source_semantic_chunk_id": semantic_chunk.get("chunk_id", ""),
                    "child_text": child_item["text"],
                    "parent_text": parent_chunk["parent_text"],
                    "word_start": child_item["word_start"],
                    "word_end": child_item["word_end"],
                    "file_name": parent_chunk["file_name"],
                    "document_summary": parent_chunk["document_summary"],
                    "source_type": parent_chunk["source_type"],
                    "chunk_type": "child",
                }
            )

    return parent_chunks, child_chunks


def build_child_embedding_text(child_chunk):
    """
    STEP 1:
    Build metadata-enriched text for child chunk embeddings.

    Important:
    We are not replacing the previous metadata anchoring version.
    This is a new parent-child retrieval version.

    The embedding model will later receive:
    - file name
    - document summary
    - child text

    But the LLM will later receive:
    - parent text
    """

    file_name = child_chunk.get("file_name", "")
    document_summary = child_chunk.get("document_summary", "")
    child_text = child_chunk.get("child_text", "")

    embedding_text = f"""
    File Name: {file_name}

    Document Summary:
    {document_summary}

    Searchable Child Chunk:
    {child_text}
    """.strip()

    return embedding_text

