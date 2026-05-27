import re #regular expressions

def clean_text(text: str) -> str:
    """
    Clean extracted document text without destroying readability.

    PDF extraction often creates:
    - too many spaces
    - too many blank lines
    - broken formatting

    This function keeps the text readable while removing noise.
    """
    if not text:
        return ""
    
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    return text.strip()

