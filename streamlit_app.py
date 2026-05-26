from numpy import percentile
import streamlit as st

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

    quetion = st.text_area(
        "Enter a complex question",
        height =140,
        placeholder="Example: What are the main risks in this document and how can they be reduced?",
    )

    st.divider()

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


# status box
if st.session_state.uploaded_file_name:
    st.success(f"Uploaded file detected: {st.session_state.uploaded_file_name}")
else:
    st.info("Upload a PDF or TXT document from the sidebar to begin.")


if run_button:
    st.warning(
        "The UI button works, but backend analysis is not connected yet. "
        "Document extraction will be added in the next step."
    )


#Create three main tabs

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

    st.info(
        "Coming next: sentence extraction, sentence embeddings, cosine distance, "
        "semantic breakpoints, and chunk visualization."
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
    st.info(
        "Coming next: context used by both approaches, token comparison, "
        "and retrieval quality explanation."
    )

