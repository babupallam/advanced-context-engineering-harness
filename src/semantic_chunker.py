import re

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