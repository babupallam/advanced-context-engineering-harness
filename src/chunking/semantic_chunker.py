import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def split_into_sentences(text):
    # output a list of dictionaries
    sentences = []
    # if input-text is none, return an empty list
    if not text:
        return sentences
    
    #### using regular expressions to split the text into sentences
    # define pattern for sentence 
    """
    This regex finds sentence-like units.

    It looks for:
    - text ending with . ? !
    - followed by whitespace
    - then starts the next sentence

    Example:
    - "Hello. How are you?" -> ["Hello.", "How are you?"]
    - "Hello. How are you? I am fine. Thank you." -> ["Hello.", "How are you?", "I am fine.", "Thank you."]

    Future:
    - replace this with a more sophisticated sentence splitter like NLTK or spaCy
    """
    sentence_pattern = r"[^.!?]+[.!?]+|[^.!?]+$"

    # find all matches of the pattern in the text
    matches = re.finditer(
        sentence_pattern, # the pattern to search for
        text, # the text to search
        flags=re.MULTILINE # this flag allows the pattern to match across multiple lines
    )

    sentence_id = 1

    for sentence in matches:
        """
        .group() (with no args) extracts the full matched text as a string 
        from the regex match object so it can be stripped of whitespace.

        Example: match.group(1)
        - Extracts only the first specific captured group (defined by () in the regex).
        - The return type is <class 'str'>.
        - Example: If regex is r"(\w+\.)", "Hello. How are you?" -> "Hello."

        Example: match.group() (or match.group(0))
        - Extracts the entire matched text exactly as the regex found it (including periods).
        - The return type is <class 'str'>.
        - Example: "Hello. How are you?" -> "Hello. How are you?"
        """
        sentence_text = sentence.group().strip() # get the text of the sentence and strip whitespace
        # if pdf extraction creates very small sentencece like "Hello." or "Hello", skip it
        if(len(sentence_text) < 10):
            continue

        # add the sentence to the list of dictionaries
        sentences.append({
            "sentence_id": sentence_id,
            "text": sentence_text,
            "char_start": text.find(sentence_text),
            "char_end": text.find(sentence_text) + len(sentence_text),
        })
        sentence_id += 1


    # return the list of dictionaries
    return sentences




def calculate_sentence_distances(sentences, embeddings):
    """
    STEP 1:
    Calculate semantic distance between neighbouring sentences.

    Example:
    - Compare sentence 1 with sentence 2
    - Compare sentence 2 with sentence 3
    - Compare sentence 3 with sentence 4

    Why:
    If two neighbouring sentences are similar, the topic is probably continuing.
    If two neighbouring sentences are very different, the topic may have changed.

    cosine_similarity:
    Value close to 1 means very similar.
    Value close to 0 means less similar.

    cosine_distance:
    distance = 1 - similarity

    Higher distance = bigger meaning shift.
    """

    distances = []

    if len(sentences) < 2:
        return distances
    
    for index in range(len(sentences) - 1):
        current_embedding = embeddings[index].reshape(1, -1) # reshape to 1D array
        next_embedding = embeddings[index + 1].reshape(1, -1) # reshape to 1D array
        similarity = cosine_similarity(current_embedding, next_embedding)[0][0] #[0][0] because it returns a 2D array
        
        distance = 1 - similarity

        distances.append(
            {
                "sentence_id_current": sentences[index]["sentence_id"],
                "sentence_id_next": sentences[index + 1]["sentence_id"],
                "cosine_similarity": round(float(similarity), 4),
                "cosine_distance": round(float(distance), 4),
                "current_sentence_preview": sentences[index]["text"][:120],
                "next_sentence_preview": sentences[index + 1]["text"][:120],
            }
        )

    # return the list of dictionaries
    #structure of distances:
    #[
    #    {
    #        "sentence_id_current": 1,
    #        "sentence_id_next": 2,
    #        "cosine_similarity": 0.95,
    #        "cosine_distance": 0.05,
    #        "current_sentence_preview": "This is the current sentence.",
    #        "next_sentence_preview": "This is the next sentence.",
    #    }
    #]
    return distances


def find_semantic_breakpoints(distances, mode="percentile"):
    """
    STEP 1:
    Find sentence positions where the topic may have changed.

    distances:
    This is the list created by calculate_sentence_distances().
    Each item contains cosine_distance between two neighbouring sentences.

    mode:
    This controls how we decide what counts as a large meaning shift.

    Available modes:
    1. percentile
       - Uses a high percentile cutoff.
       - Example: distances above the 85th percentile become breakpoints.

    2. standard_deviation
       - Uses mean distance + 1 standard deviation.
       - This marks unusually large jumps as breakpoints.

    Return:
    A list of sentence IDs where chunks should be split.
    """
    if not distances:
        return []

    distance_values = np.array(
        [
            item["cosine_distance"] for item in distances
        ]
    )
    if mode == "percentile":
        threshold = np.percentile(distance_values, 85) #85 is the 85th percentile of the distance values
    elif mode == "standard_deviation":
        threshold = np.mean(distance_values) + np.std(distance_values)
    else:
        raise ValueError(f"Invalid mode: {mode}, Invalid threshold mode. Use 'percentile' or 'standard_deviation'.")
    
    breakpoints = []

    for item in distances:
        """
        STEP 2:
        If the distance is greater than or equal to the threshold,
        we treat the gap between current sentence and next sentence
        as a possible semantic boundary.

        Example:
        If sentence_id_current = 8 has a large distance to sentence 9,
        then we end the current chunk at sentence 8.
        """
        if item["cosine_distance"] >= threshold:
            breakpoints.append(item["sentence_id_current"])

    return breakpoints


def build_semantic_chunks(sentences, breakpoints):
    """
    STEP 1:
    Build chunks using semantic breakpoints.

    sentences:
    List of sentence dictionaries.

    breakpoints:
    Sentence IDs where a chunk should end.

    Example:
    Sentences:
    1, 2, 3, 4, 5, 6

    Breakpoints:
    3

    Result:
    Chunk 1 = sentences 1 to 3
    Chunk 2 = sentences 4 to 6

    Return:
    A list of semantic chunk dictionaries.
    """
    semantic_chunks = []

    #structure of sentences:
    #[
    #    {
    #        "sentence_id": 1,
    #        "text": "This is the first sentence.",
    #        "char_start": 0,
    #        "char_end": 20,
    #    }
    #]

    #structure of breakpoints:
    #[
    #    3,
    #    6,
    #]

    if not sentences:
        return semantic_chunks
    
    breakpoint_set = set(breakpoints)
    
    current_chunk_sentences = []
    chunk_id = 1

    for sentence in sentences:
        current_chunk_sentences.append(sentence)

        """
        STEP 2:
        If the sentence is a breakpoint, we end the current chunk
        and start a new chunk.
        """
        if sentence["sentence_id"] in breakpoint_set:
            chunk_text = " ".join(
                item["text"] for item in current_chunk_sentences
            ).strip()

            semantic_chunks.append(
                {
                    "chunk_id": f"semantic_{chunk_id}",
                    "text": chunk_text,
                    "sentence_start": current_chunk_sentences[0]["sentence_id"],
                    "sentence_end": current_chunk_sentences[-1]["sentence_id"],
                    "char_start": current_chunk_sentences[0]["char_start"],
                    "char_end": current_chunk_sentences[-1]["char_end"],
                    "chunk_type": "semantic",
                }
            )

            chunk_id += 1
            #reset the current chunk sentences
            current_chunk_sentences = []

    """
    STEP 3:
    If there are remaining sentences, add the last chunk.
    """
    # after the loop, the current chunk sentences will contain the last chunk of sentences
    # current_chunk_sentences is empty if there are no breakpoints
    if current_chunk_sentences:
        chunk_text = " ".join(
            item["text"] for item in current_chunk_sentences
        ).strip()

        semantic_chunks.append(
            {
                "chunk_id": f"semantic_{chunk_id}",
                "text": chunk_text,
                "sentence_start": current_chunk_sentences[0]["sentence_id"],
                "sentence_end": current_chunk_sentences[-1]["sentence_id"],
                "char_start": current_chunk_sentences[0]["char_start"],
                "char_end": current_chunk_sentences[-1]["char_end"],
                "chunk_type": "semantic",
            }
        )
    
    #structure of semantic_chunks:
    #[
    #    {
    #        "chunk_id": "semantic_1",
    #        "text": "This is the first sentence. This is the second sentence.",
    #        "sentence_start": 1,
    #        "sentence_end": 2,
    #        "char_start": 0,
    #        "char_end": 40,
    #        "chunk_type": "semantic",
    #    }
    #]
    return semantic_chunks


