def create_document_metadata(file_name, full_text):
    """
    STEP 1:
    Create simple metadata for the uploaded document.

    Metadata means extra information about the source document.

    Example:
    Instead of embedding only:
        "Revenue increased by 12%."

    We can attach:
        File name: annual_report.pdf
        Summary: This document discusses company performance...
        Total words: 5000

    This gives each chunk more context.
    """
    if not full_text:
        return {
            "file_name": file_name,
            "document_summary": "",
            "total_characters": 0,
            "total_words": 0,
            "source_type": "uploaded_file",
        }
    
    #split the text into words
    words = full_text.split()

    """
    Create a simple extracted summary of the document.

    For now, we use the first meaningful sentences as the summary.
    Later we can replace this with a more sophisticated summary generation model.
    """

    summary_words = words[:80]
    document_summary = " ".join(summary_words)

    return {
        "file_name": file_name,
        "document_summary": document_summary,
        "total_characters": len(full_text),
        "total_words": len(words),
        "source_type": "uploaded_file",
    }



def attach_metadata_to_chunks(chunks, metadata):
    """
    STEP 1:
    Attach document metadata to every chunk.

    We do not remove the original chunk fields.
    We only add extra fields.

    Each chunk receives:
    - file_name
    - document_summary
    - source_type
    - total_document_words
    """

    anchored_chunks = []

    for chunk in chunks:
        updated_chunk = chunk.copy()
        updated_chunk["file_name"] = metadata["file_name"]
        updated_chunk["document_summary"] = metadata["document_summary"]
        updated_chunk["source_type"] = metadata["source_type"]
        updated_chunk["total_words"] = metadata["total_words"]
        anchored_chunks.append(updated_chunk)
    
    return anchored_chunks


def build_embedding_texts(chunk):
    """
    STEP 1:
    Build the text that will later be sent to the embedding model.

    Important:
    The user sees clean chunk text in the UI.
    But the embedding model receives chunk text plus metadata.

    Why:
    This helps the vector represent both:
    - local chunk meaning
    - global document context

    """

    file_name = chunk.get("file_name", "")
    document_summary = chunk.get("document_summary", "")
    chunk_text = chunk.get("text", "")

    embedding_text = f"""
    File name: {file_name}
    Document summary: {document_summary}
    Chunk text: {chunk_text}
    """.strip()
    return embedding_text

