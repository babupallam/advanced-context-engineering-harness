import re
from sklearn.metrics.pairwise import cosine_similarity


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
    return distances


