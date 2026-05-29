def naive_chunk_text(text, chunk_size=1000, overlap=150):
    """
    STEP 1:
    Split the document using fixed character length.

    This is called "naive chunking" because it does not understand meaning.
    It simply cuts the text every fixed number of characters.

    Example:
    - Chunk 1: character 0 to 1000
    - Chunk 2: character 850 to 1850
    - Chunk 3: character 1700 to 2700

    The overlap helps reduce information loss near boundaries.

    Parameters:
    text:
        Full cleaned document text.

    chunk_size:
        Number of characters in each chunk.

    overlap:
        Number of characters repeated from the previous chunk.

    Returns:
        A list of dictionaries.
        Each dictionary represents one chunk.
    """
    # creating a list to store the chunks
    chunks = []

    if not text:
        return chunks

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    start_char = 0
    chunk_id = 1
    text_length = len(text)

    # iterating through the text in chunks of chunk_size
    while(start_char < text_length):
        end_char = min(start_char + chunk_size, text_length)
        chunk_text = text[start_char:end_char].strip()
        if chunk_text:
            # adding the chunk to the list
            chunks.append({
                "chunk_id": f"naive_{chunk_id}",
                "text": chunk_text,
                "start_character": start_char,
                "end_character": end_char,
                "character_count": len(chunk_text),
                "chunk_type": "naive",
            })
        # updating the start character for the next chunk
        start_char = start_char + chunk_size - overlap
        chunk_id += 1

    # returning the list of chunks
    return chunks