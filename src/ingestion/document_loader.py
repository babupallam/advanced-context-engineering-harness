# fitz is the PyMuPDF library for PDF processing
import fitz
import pandas as pd

def extract_text_from_pdf(uploaded_file):
    """
    STEP 1:
    Extract text from a PDF uploaded through Streamlit.

    uploaded_file:
    This is the file object returned by st.file_uploader.

    fitz:
    This comes from PyMuPDF.
    It allows Python to open and read PDF files.

    Why we use uploaded_file.read():
    Streamlit gives the uploaded file as bytes.
    PyMuPDF can open those bytes directly.
    """
    pdf_bytes = uploaded_file.read()
    document = fitz.open(stream=pdf_bytes, filetype="pdf"
    )

    extracted_pages = []

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text()
        if page_text.strip():
            extracted_pages.append(
                f"\n\n--- Page {page_number} ---\n{page_text.strip()}"

            )

    document.close() #close the document to free resources

    # \n is used to separate the pages
    #for ex: print(" ".join([1,2,3])) will print "1 2 3"
    return "\n".join(extracted_pages)



def extract_text_from_txt(uploaded_file):
    """
    STEP 1:
    Extract text from a TXT file uploaded through Streamlit.

    TXT files can have different encodings.
    We try UTF-8 first because it is the most common.
    If that fails, we use latin-1 as a fallback.
    """
    text_bytes = uploaded_file.read()
    
    try:
        text = text_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = text_bytes.decode("latin-1")
    return text

def extract_text_from_csv(uploaded_file):
    """
    STEP 1:
    Extract text from a CSV file uploaded through Streamlit.

    CSV files are read using pandas.
    """
    df = pd.read_csv(uploaded_file)
    lines = []
    for row_index, row in df.iterrows():
        row_parts = []
        for column_name, value in row.items():
            row_parts.append(f"{column_name}: {value}")
        lines.append(f"Row {row_index + 1}: " + " | ".join(row_parts))
    
    return "\n".join(lines)


def load_uploaded_document(uploaded_file):
    """
    STEP 1:
    Detect the uploaded file type.

    STEP 2:
    Send the file to the correct extraction function.

    STEP 3:
    Return extracted text.

    This keeps streamlit_app.py clean because the app does not
    need to know the internal details of PDF/TXT extraction.
    """
    if uploaded_file is None:
        return None
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    elif file_name.endswith(".csv"):
        return extract_text_from_csv(uploaded_file)
    else:
        return ValueError(f"Unsupported file type: {file_name}")

