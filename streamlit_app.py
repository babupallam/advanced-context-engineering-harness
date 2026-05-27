from numpy import percentile
import streamlit as st

from src.document_loader import load_uploaded_document
from src.text_cleaner import clean_text
from src.naive_chunker import naive_chunk_text
from src.semantic_chunker import split_into_sentences



st.set_page_config(
    page_title="Advanced Context Engineering Harness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#session state initialization
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""

if "naive_chunks" not in st.session_state:
    st.session_state.naive_chunks = []

if "semantic_chunks" not in st.session_state:
    st.session_state.semantic_chunks = []

if "sentences" not in st.session_state:
    st.session_state.sentences =[]

if "row_vector_results" not in st.session_state:
    st.session_state.row_vector_results = []

if "reranked_results" not in st.session_state:
    st.session_state.reranked_results = []

if "naive_answer" not in st.session_state:
    st.session_state.naive_answer = ""

if "engineered_answer" not in st.session_state:
    st.session_state.engineered_answer = ""

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "full_document_tokens": 0,#total tokens in full document
        "naive_context_tokens": 0,#total tokens in naive context
        "engineered_context_tokens": 0,#total tokens in engineered context
        "tokens_saved_percent": 0,#percentage of tokens saved by engineered context
        "context_reduction_percent": 0,#percentage of context reduction by engineered context
    }

#title and description
st.title("Advanced Context Engineering Harness")

# caption under title
st.caption(
    "Compare naive RAG with engineered context retrieval using semantic chunking, "
    "metadata anchoring, parent-child retrieval, re-ranking, and token-efficiency metrics."
)

#sidebar 

with st.sidebar:
    st.header("Input controls")

    uploaded_file = st.file_uploader(
        "Upload a PDF, TXT, or CSV file", 
        type=["pdf", "txt", "csv"]
    )

    question = st.text_area(
        "Enter a complex question",
        height =140,
        placeholder="Example: What are the main risks in this document and how can they be reduced?",
    )

    st.divider()

    st.subheader("Naive Chunk Settings")

    naive_chunk_size = st.number_input(
        "Naive chunk size",
        min_value=300,
        max_value=3000,
        value=1000,
        step=100
    )

    naive_chunk_overlap = st.number_input(
        "Naive chunk overlap",
        min_value=0,
        max_value=500,
        value=150,
        step=50
    )


    st.subheader("Retrieval Settings")

    embedding_model_name = st.selectbox(
        "Embedding Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
    )
    
    reranker_model_name = st.selectbox(
         "Re-ranker model",
        [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2"
        ]
    )
    
    threshold_mode = st.selectbox(
        "Semantic threshold mode",
        [
            "prcentile",
            "standard_deviation"
        ]
   )

    raw_top_k = st.number_input(
        "Top-k row vector results",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )
    final_top_n = st.number_input(
        "Final top-n results",
        min_value=1,
        max_value=50,
        value=4,
        step=1
    )
    # this button is used to run the context engineering analysis
    run_button = st.button(
        "Run Context Engineering Analysis  ",
        type="primary",
        use_container_width=True
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name
        

# Document loading and cleaning

if uploaded_file is not None:
    try:
        document_text = load_uploaded_document(uploaded_file)
        cleaned_text = clean_text(document_text)

        st.session_state.document_text = cleaned_text
        st.session_state.uploaded_file_name = uploaded_file.name

        # Generate naive chunks immediately after text extraction and cleaning
        st.session_state.naive_chunks = naive_chunk_text(
            text = st.session_state.document_text,
            chunk_size= naive_chunk_size,
            overlap= naive_chunk_overlap
        )
        st.session_state.sentences = split_into_sentences(st.session_state.document_text)

    except Exception as e:
        st.error(f"Document extraction or chunking failed: {e}")
        st.session_state.document_text = ""
        st.session_state.uploaded_file_name = ""

#Basic document stats
document_character_count = len(st.session_state.document_text) if st.session_state.document_text else 0
document_word_count = len(st.session_state.document_text.split()) if st.session_state.document_text else 0



st.subheader("Context Efficiency Metrics")    

matric_col1, matric_col2, matric_col3, matric_col4, matric_col5 = st.columns(5)

with matric_col1:
    st.metric(
        "Full Document Tokens",
        st.session_state.metrics["full_document_tokens"]
    )

with matric_col2:
    st.metric(
        "Naive Context Tokens",
        st.session_state.metrics["naive_context_tokens"]
    )

with matric_col3:
    st.metric(
        "Engineered Context Tokens",
        st.session_state.metrics["engineered_context_tokens"]
    )

with matric_col4:
    st.metric(
        "Tokens Saved Percent",
        st.session_state.metrics["tokens_saved_percent"]
    )

with matric_col5:
    st.metric(
        "Context Reduction Percent",
        st.session_state.metrics["context_reduction_percent"]
    )


# status box - document status
if st.session_state.uploaded_file_name:
    st.success(f"Uploaded and extracted: {st.session_state.uploaded_file_name}")
    state_col1, state_col2 = st.columns(2)
    with state_col1:
        st.metric("Document Character Count", document_character_count)
    with state_col2:
        st.metric("Document Word Count", document_word_count)
    with st.expander("Document Text Preview"):
        st.text_area(
            "Extracted Document Preview", 
            st.session_state.document_text[:500] + "...", 
            height=200, 
            disabled=True
        )

else:
    st.info("Upload a PDF or TXT document from the sidebar to begin.")


if run_button:
    if not st.session_state.document_text:
        st.warning("Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question first.")
    else:
        st.warning("Document extraction is working. The next step will add naive chunking.")
            

# Three tabs for displaying different aspects of the context engineering process
tab_1, tab_2, tab_3 = st.tabs(
    [
        "Semantic Chunk Map",
        "Re-ranker Analytics",
        "Naive RAG vs Engineered Context"
    ]
)

# ============================================================
# TAB 1: SEMANTIC CHUNK MAP
# ============================================================
# This tab will later show:
# - sentence count
# - naive chunk count
# - semantic chunk count
# - cosine distance table
# - semantic breakpoint chart
# - final semantic chunks

with tab_1:
    st.header("Semantic Chunk Map")

    st.write(
        "This tab will show how the document is split based on meaning shifts "
        "instead of fixed character counts."
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Total Sentences", len(st.session_state.sentences))

    with col_b:
        st.metric("Naive Chunk Count", len(st.session_state.naive_chunks))

    with col_c:
        st.metric("Semantic Chunk Count", len(st.session_state.semantic_chunks))

    st.divider()
    st.header("Sentence Splitting Preview")
    if st.session_state.sentences:
        sentence_preview = st.session_state.sentences[:20]
        st.dataframe(sentence_preview,
        use_container_width=True,
        hide_index=True,
        )

        with st.expander("View all extracted sentences"):
            st.dataframe(st.session_state.sentences,
            use_container_width=True,
            hide_index=True,
            )
    else:
        st.info("No sentences available.")
        st.info("Upload a document to generate sentence-level units.")

    st.info(
        "Coming next: sentence embeddings, cosine distance, semantic breakpoints, "
        "and chunk visualization."
    )



# ============================================================
# TAB 2: RE-RANKER ANALYTICS
# ============================================================
# This tab will later show:
# - top 10 raw vector search hits
# - vector similarity scores
# - cross-encoder re-ranker scores
# - final top 3 selected chunks

with tab_2:
    st.header("Re-ranker Analytics")
    st.write("This tab will compare raw vector retrieval results against deeper "
        "cross-encoder re-ranking results.")
    st.info(
        "Coming next: re-ranker model selection, re-ranker threshold mode, "
        "re-ranker top-k results, and re-ranker final top-n results."
    )



# ============================================================
# TAB 3: NAIVE RAG VS ENGINEERED CONTEXT
# ============================================================
# This tab will later show:
# - naive answer
# - engineered context answer
# - context used by both approaches
# - token comparison
# - retrieval quality explanation

with tab_3:
    st.header("Naive RAG vs Engineered Context")
    left_col, right_col = st.columns(2)
    with left_col:
        st. subheader("Naive RAG Answer")
        st.write( 
            "The naive answer will appear here after fixed-size chunking, "
            "vector search, and LLM generation are added."
        )
    with right_col:
        st. subheader("Engineered Context Answer")
        st.write( 
            "The engineered context answer will appear here after semantic chunking, "
            "parent-child retrieval, re-ranking, and compact context building are added."
        )
    st.divider()
    with st.expander("Naive Chunks Preview"):
        if st.session_state.naive_chunks:
            for chunk in st.session_state.naive_chunks:
                st.markdown(f"**Chunk {chunk['chunk_id']}**")
                st.caption(f"Start: {chunk['start_character']}, End: {chunk['end_character']}, Characters: {chunk['character_count']}")
                st.text_area(
                    "Chunk Text",
                    chunk['text'],
                    height=100,
                    disabled=True
                )
                st.divider()
        else:
            st.caption("No naive chunks available.")            
            st.info("Upload a document to generate naive chunks.")

    st.info(
        "Coming next: sentence splitting for semantic chunking."
    )



