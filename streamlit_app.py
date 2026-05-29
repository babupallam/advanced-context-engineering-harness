import logging
import os
import warnings

# Suppress Hugging Face advisory noise when Streamlit inspects lazy transformers submodules.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"Accessing `__path__` from")

import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ingestion.document_loader import load_uploaded_document
from src.ingestion.text_cleaner import clean_text
from src.chunking.naive_chunker import naive_chunk_text
from src.retrieval.embeddings import load_embedding_model, embed_texts
from src.chunking.semantic_chunker import (
    split_into_sentences,
    calculate_sentence_distances,
    find_semantic_breakpoints,
    build_semantic_chunks
)
from src.metadata.metadata_anchor import (
    create_document_metadata,
    attach_metadata_to_chunks,
    build_embedding_texts,
)
from src.retrieval.parent_child import create_parent_child_chunks, build_child_embedding_text
from src.retrieval.vector_store import build_vector_index, retrieve_top_k
from src.retrieval.reranker import (
    load_reranker_model,
    rerank_results,
    get_selected_reranked_results,
)

from src.generation.context_builder import build_naive_context, build_engineered_context
from src.metrics.metrics import (
    calculate_context_metrics,
    calculate_llm_call_token_estimates,
    count_tokens,
)
from src.config.model_config import (
    DEFAULT_MODEL_DISPLAY_NAME,
    get_provider_model_id,
    has_placeholder_model_ids,
)
from src.config.pipeline_config import get_pipeline_defaults
from src.generation.llm_client import generate_answer
from src.evaluation.process_logger import (
    create_uploaded_file_log_name,
    get_current_timestamp,
    log_process_event,
)
from src.evaluation.similarity_analysis import calculate_chunk_similarity_matrix
from src.ui.ui_components import (
    show_pipeline_summary,
    show_metric_cards,
    show_context_difference_metrics,
    show_chunk_card,
    show_info_box,
    show_warning_for_large_document,
    show_token_metric_disclaimer,
    show_pipeline_defaults_expander,
    show_formula_hint,
    show_llm_token_estimate_column,
    show_context_with_chunk_expanders,
    show_processing_time_summary,
)


def build_analysis_log_event(
    *,
    run_id,
    event_type,
    question,
    llm_model_display_name,
    embedding_model_name,
    reranker_model_name,
    threshold_mode,
    naive_chunk_size,
    naive_chunk_overlap,
    raw_top_k,
    final_top_n,
    child_max_words,
    child_overlap_words,
    naive_context_max_chunks,
    status,
    error_message="",
    total_processing_seconds=0,
):
    """Build a CSV log row dict from current session state and run settings."""

    return {
        "timestamp": get_current_timestamp(),
        "run_id": run_id,
        "event_type": event_type,
        "uploaded_file_name": st.session_state.uploaded_file_name,
        "uploaded_file_extension": st.session_state.get("uploaded_file_extension", ""),
        "stored_file_log_name": st.session_state.get("stored_file_log_name", ""),
        "question": question,
        "selected_llm_model": llm_model_display_name,
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "semantic_threshold_mode": threshold_mode,
        "naive_chunk_size": naive_chunk_size,
        "naive_chunk_overlap": naive_chunk_overlap,
        "raw_top_k": raw_top_k,
        "final_top_n": final_top_n,
        "child_max_words": child_max_words,
        "child_overlap_words": child_overlap_words,
        "naive_context_max_chunks": naive_context_max_chunks,
        "full_document_tokens": st.session_state.metrics.get("full_document_tokens", 0),
        "naive_context_tokens": st.session_state.metrics.get("naive_context_tokens", 0),
        "engineered_context_tokens": st.session_state.metrics.get(
            "engineered_context_tokens", 0
        ),
        "context_size_difference": st.session_state.metrics.get(
            "context_size_difference", 0
        ),
        "context_change_percent": st.session_state.metrics.get("context_change_percent", 0),
        "context_reduction_percent": st.session_state.metrics.get(
            "context_reduction_percent", 0
        ),
        "naive_estimated_input_tokens": st.session_state.naive_llm_token_estimate.get(
            "estimated_input_tokens", 0
        ),
        "naive_estimated_output_tokens": st.session_state.naive_llm_token_estimate.get(
            "estimated_output_tokens", 0
        ),
        "naive_estimated_total_tokens": st.session_state.naive_llm_token_estimate.get(
            "estimated_total_tokens", 0
        ),
        "engineered_estimated_input_tokens": st.session_state.engineered_llm_token_estimate.get(
            "estimated_input_tokens", 0
        ),
        "engineered_estimated_output_tokens": st.session_state.engineered_llm_token_estimate.get(
            "estimated_output_tokens", 0
        ),
        "engineered_estimated_total_tokens": st.session_state.engineered_llm_token_estimate.get(
            "estimated_total_tokens", 0
        ),
        "document_processing_seconds": st.session_state.processing_times.get(
            "document_processing_seconds", 0
        ),
        "chunking_seconds": st.session_state.processing_times.get("chunking_seconds", 0),
        "embedding_seconds": st.session_state.processing_times.get("embedding_seconds", 0),
        "vector_search_seconds": st.session_state.processing_times.get(
            "vector_search_seconds", 0
        ),
        "reranking_seconds": st.session_state.processing_times.get("reranking_seconds", 0),
        "naive_llm_seconds": st.session_state.processing_times.get("naive_llm_seconds", 0),
        "engineered_llm_seconds": st.session_state.processing_times.get(
            "engineered_llm_seconds", 0
        ),
        "total_processing_seconds": total_processing_seconds,
        "status": status,
        "error_message": error_message,
    }


st.set_page_config(
    page_title="Advanced Context Engineering Harness",
    page_icon="assets/context-engineering.png",
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
        "full_document_tokens": 0,
        "naive_context_tokens": 0,
        "engineered_context_tokens": 0,
        "context_size_difference": 0,
        "context_change_percent": 0,
        "context_reduction_percent": 0,
    }

if "selected_llm_model_name" not in st.session_state:
    st.session_state.selected_llm_model_name = ""

if "sentence_embeddings" not in st.session_state:
    st.session_state.sentence_embeddings = []

if "sentence_distances" not in st.session_state:
    st.session_state.sentence_distances = []

if "semantic_breakpoints" not in st.session_state:
    st.session_state.semantic_breakpoints = []

if "semantic_chunks" not in st.session_state:
    st.session_state.semantic_chunks = []

if "document_metadata" not in st.session_state:
    st.session_state.document_metadata = {}
if "parent_chunks" not in st.session_state:
    st.session_state.parent_chunks = []
if "child_chunks" not in st.session_state:
    st.session_state.child_chunks = []
if "vector_index" not in st.session_state:
    st.session_state.vector_index = {}
if "raw_vector_results" not in st.session_state:
    st.session_state.raw_vector_results = []


if "naive_context" not in st.session_state:
    st.session_state.naive_context = ""
if "engineered_context" not in st.session_state:
    st.session_state.engineered_context = ""

if "naive_llm_token_estimate" not in st.session_state:
    st.session_state.naive_llm_token_estimate = {}

if "engineered_llm_token_estimate" not in st.session_state:
    st.session_state.engineered_llm_token_estimate = {}

if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = ""

if "processing_times" not in st.session_state:
    st.session_state.processing_times = {}

if "stored_file_log_name" not in st.session_state:
    st.session_state.stored_file_log_name = ""

if "uploaded_file_extension" not in st.session_state:
    st.session_state.uploaded_file_extension = ""

if "chunk_similarity_matrix" not in st.session_state:
    st.session_state.chunk_similarity_matrix = None

if "chunk_average_similarity" not in st.session_state:
    st.session_state.chunk_average_similarity = None


#cache the embedding model
@st.cache_resource
def get_cached_embedding_model(model_name: str):
    """
    Why:
    Loading ML models can be slow.
    st.cache_resource keeps the model in memory after first load.
    """

    return load_embedding_model(model_name)

#cache the re-ranker model
@st.cache_resource
def get_cached_reranker_model(model_name):
    """
    STEP 1:
    Cache the cross-encoder re-ranker model.

    Why:
    Cross-encoder models can take time to load.
    Caching prevents repeated loading on every Streamlit rerun.
    """

    return load_reranker_model(model_name)


#title and description
st.title("Advanced Context Engineering Harness")

# caption under title
st.caption(
    "Compare naive RAG with engineered context retrieval using semantic chunking, "
    "metadata anchoring, parent-child retrieval, re-ranking, and token-efficiency metrics."
)

show_pipeline_summary()

#sidebar 

with st.sidebar:
    st.header("Input controls")

    try:
        st.page_link("pages/1_History_Logs.py", label="View History Logs", icon="📊")
    except Exception:
        st.info("Open the History Logs page from the Streamlit sidebar navigation.")

    with st.expander("How this app works"):
        st.markdown(
            """
**Getting started**

1. Upload a **PDF** or **TXT** file.
2. Enter a **complex question** about the document.
3. Click **Run Context Engineering Analysis**.

**What each tab shows**

- **Tab 1 — Engineered Pipeline: Semantic Chunk Map:** Preparation stages before final retrieval (sentence splitting through parent-child structure).
- **Tab 2 — Engineered Pipeline: Retrieval and Re-ranking Analytics:** Candidate retrieval, cross-encoder re-ranking, and final evidence selection.
- **Tab 3 — Naive RAG vs Engineered Context:** Side-by-side answers, contexts, token metrics, and pipeline stage review.
            """
        )

    uploaded_file = st.file_uploader(
        "Upload a PDF, TXT, or CSV file",
        type=["pdf", "txt", "csv"],
        help="Source document for chunking, retrieval, and answer generation.",
    )

    question = st.text_area(
        "Enter a complex question",
        height=140,
        placeholder="Example: What are the main risks in this document and how can they be reduced?",
        help="Question used for vector retrieval, re-ranking, and both LLM answers.",
    )

    st.divider()

    st.subheader("Naive Chunk Settings")

    naive_chunk_size = st.number_input(
        "Naive chunk size",
        min_value=300,
        max_value=3000,
        value=1000,
        step=100,
        help="Character window size for fixed-size naive RAG chunks.",
    )

    naive_chunk_overlap = st.number_input(
        "Naive chunk overlap",
        min_value=0,
        max_value=500,
        value=150,
        step=50,
        help="Overlapping characters between adjacent naive chunks.",
    )


    st.subheader("Retrieval Settings")

    embedding_model_name = st.selectbox(
        "Embedding Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        help="Sentence embedding model for semantic distance and vector search.",
    )

    reranker_model_name = st.selectbox(
        "Re-ranker model",
        [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
        ],
        help="Cross-encoder that scores question–chunk pairs for precision ranking.",
    )

    threshold_mode = st.selectbox(
        "Semantic threshold mode",
        [
            "percentile",
            "standard_deviation",
        ],
        help="How semantic distance spikes are converted into chunk boundaries.",
    )

    raw_top_k = st.number_input(
        "Top-k row vector results",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Number of candidate chunks retrieved from vector search before re-ranking.",
    )
    final_top_n = st.number_input(
        "Final top-n results",
        min_value=1,
        max_value=50,
        value=4,
        step=1,
        help="Number of highest-ranked chunks selected after cross-encoder re-ranking.",
    )

    st.divider()

    st.subheader("Parent-Child Settings")

    child_max_words = st.number_input(
        "Child chunk max words",
        min_value=80,
        max_value=500,
        value=220,
        step=20,
        help="Maximum words allowed in each child chunk used for retrieval.",
    )

    child_overlap_words = st.number_input(
        "Child chunk overlap words",
        min_value=0,
        max_value=150,
        value=40,
        step=10,
        help="Word overlap between adjacent child chunks to reduce boundary information loss.",
    )

    st.divider()

    st.subheader("Context Building Settings")

    naive_context_max_chunks = st.number_input(
        "Naive context max chunks",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Maximum number of retrieved chunks included in the naive RAG context.",
    )

    engineered_context_max_chunks = st.number_input(
        "Engineered context max chunks",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Maximum parent chunks included in the engineered context after re-ranking.",
    )


    st.divider()

    st.subheader("LLM Provider Settings")

    llm_model_options = [
        "Trinity Mini",
        "Gemma 4 31B IT",
        "Llama 3.1 8B",
        "Meta Llama 3.3 70B Instruct",
        "GPT OSS 120B",
        "GPT OSS 20B",
        "Qwen 3 235B A22B Thinking 2507",
        "Qwen 3.5 9B",
    ]

    llm_model_display_name = st.selectbox(
        "LLM model",
        llm_model_options,
        index=llm_model_options.index(DEFAULT_MODEL_DISPLAY_NAME),
        help="Display name mapped to a provider model ID in src/config/model_config.py.",
    )

    llm_temperature = st.slider(
        "LLM temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Controls randomness. Lower values are more deterministic.",
    )

    show_pipeline_defaults_expander(get_pipeline_defaults())

    # this button is used to run the context engineering analysis
    run_button = st.button(
        "Run Context Engineering Analysis  ",
        type="primary",
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name
        

# Document loading and cleaning

if uploaded_file is not None:
    try:
        st.session_state.stored_file_log_name = create_uploaded_file_log_name(
            uploaded_file.name
        )
        st.session_state.uploaded_file_extension = Path(uploaded_file.name).suffix or ""

        document_start = time.perf_counter()
        document_text = load_uploaded_document(uploaded_file)
        cleaned_text = clean_text(document_text)
        st.session_state.processing_times["document_processing_seconds"] = round(
            time.perf_counter() - document_start,
            4,
        )

        st.session_state.document_text = cleaned_text
        st.session_state.uploaded_file_name = uploaded_file.name

        st.session_state.document_metadata = create_document_metadata(
            uploaded_file.name,
            st.session_state.document_text,
        )

        chunking_start = time.perf_counter()
        naive_chunks = naive_chunk_text(
            text=st.session_state.document_text,
            chunk_size=naive_chunk_size,
            overlap=naive_chunk_overlap,
        )
        st.session_state.naive_chunks = attach_metadata_to_chunks(
            naive_chunks, st.session_state.document_metadata
        )

        st.session_state.sentences = split_into_sentences(st.session_state.document_text)

        if st.session_state.sentences:
            sentence_texts = [sentence["text"] for sentence in st.session_state.sentences]
            embedding_model = get_cached_embedding_model(embedding_model_name)
            st.session_state.sentence_embeddings = embed_texts(
                sentence_texts, embedding_model
            )
            st.session_state.sentence_distances = calculate_sentence_distances(
                st.session_state.sentences,
                st.session_state.sentence_embeddings,
            )
            st.session_state.semantic_breakpoints = find_semantic_breakpoints(
                distances=st.session_state.sentence_distances,
                mode=threshold_mode,
            )

            semantic_chunks = build_semantic_chunks(
                sentences=st.session_state.sentences,
                breakpoints=st.session_state.semantic_breakpoints,
            )
            st.session_state.semantic_chunks = attach_metadata_to_chunks(
                semantic_chunks, st.session_state.document_metadata
            )

            parent_chunks, child_chunks = create_parent_child_chunks(
                semantic_chunks=st.session_state.semantic_chunks,
                child_max_words=child_max_words,
                child_overlap_words=child_overlap_words,
            )
            st.session_state.parent_chunks = parent_chunks
            st.session_state.child_chunks = child_chunks
        else:
            st.session_state.semantic_breakpoints = []
            st.session_state.semantic_chunks = []
            st.session_state.parent_chunks = []
            st.session_state.child_chunks = []

        st.session_state.processing_times["chunking_seconds"] = round(
            time.perf_counter() - chunking_start,
            4,
        )

        if st.session_state.sentences and st.session_state.child_chunks:
            embedding_start = time.perf_counter()
            embedding_model = get_cached_embedding_model(embedding_model_name)

            st.session_state.vector_index = build_vector_index(
                child_chunks=st.session_state.child_chunks,
                embedding_model=embedding_model,
            )

            if st.session_state.semantic_chunks:
                similarity_matrix, average_similarity_df = calculate_chunk_similarity_matrix(
                    st.session_state.semantic_chunks,
                    embedding_model,
                )
                st.session_state.chunk_similarity_matrix = similarity_matrix
                st.session_state.chunk_average_similarity = average_similarity_df
            else:
                st.session_state.chunk_similarity_matrix = None
                st.session_state.chunk_average_similarity = None

            st.session_state.processing_times["embedding_seconds"] = round(
                time.perf_counter() - embedding_start,
                4,
            )
        else:
            st.session_state.vector_index = {}
            st.session_state.chunk_similarity_matrix = None
            st.session_state.chunk_average_similarity = None
            st.session_state.processing_times["embedding_seconds"] = 0

    except Exception as e:
        st.error(f"Document extraction or chunking failed: {e}")
        st.session_state.document_text = ""
        st.session_state.uploaded_file_name = ""
        st.session_state.chunk_similarity_matrix = None
        st.session_state.chunk_average_similarity = None

#Basic document stats
document_character_count = len(st.session_state.document_text) if st.session_state.document_text else 0
document_word_count = len(st.session_state.document_text.split()) if st.session_state.document_text else 0


# status box - document status
if st.session_state.uploaded_file_name:
    st.success(f"Uploaded and extracted: {st.session_state.uploaded_file_name}")
    show_warning_for_large_document(st.session_state.document_text)
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
    elif not st.session_state.child_chunks:
        st.warning(
            "Child chunks are not available yet. Upload a document and make sure "
            "semantic chunking and parent-child retrieval were created."
        )  
    elif not st.session_state.vector_index:
        st.warning(
            "Vector index is not available yet. Re-upload the document to build the index."
        )
    else:
        run_id = str(uuid.uuid4())
        st.session_state.current_run_id = run_id
        total_start_time = time.perf_counter()

        log_settings = {
            "question": question,
            "llm_model_display_name": llm_model_display_name,
            "embedding_model_name": embedding_model_name,
            "reranker_model_name": reranker_model_name,
            "threshold_mode": threshold_mode,
            "naive_chunk_size": naive_chunk_size,
            "naive_chunk_overlap": naive_chunk_overlap,
            "raw_top_k": raw_top_k,
            "final_top_n": final_top_n,
            "child_max_words": child_max_words,
            "child_overlap_words": child_overlap_words,
            "naive_context_max_chunks": naive_context_max_chunks,
        }

        try:
            embedding_model = get_cached_embedding_model(embedding_model_name)

            vector_search_start = time.perf_counter()
            st.session_state.raw_vector_results = retrieve_top_k(
                query=question,
                child_chunks=st.session_state.child_chunks,
                index=st.session_state.vector_index,
                embedding_model=embedding_model,
                top_k=raw_top_k,
            )
            st.session_state.processing_times["vector_search_seconds"] = round(
                time.perf_counter() - vector_search_start,
                4,
            )

            reranking_start = time.perf_counter()
            reranker_model = get_cached_reranker_model(reranker_model_name)
            st.session_state.reranked_results = rerank_results(
                query=question,
                raw_results=st.session_state.raw_vector_results,
                reranker_model=reranker_model,
                top_n=final_top_n,
            )
            st.session_state.processing_times["reranking_seconds"] = round(
                time.perf_counter() - reranking_start,
                4,
            )

            st.session_state.naive_context = build_naive_context(
                raw_vector_results=st.session_state.raw_vector_results,
                max_chunks=naive_context_max_chunks,
            )

            st.session_state.engineered_context = build_engineered_context(
                reranked_results=st.session_state.reranked_results,
            )

            llm_api_key = st.secrets.get("LLM_API_KEY", "")
            llm_base_url = st.secrets.get("LLM_BASE_URL", "")

            if not llm_api_key:
                raise ValueError("LLM_API_KEY is missing from Streamlit secrets.")

            if not llm_base_url:
                raise ValueError("LLM_BASE_URL is missing from Streamlit secrets.")

            llm_model_name = get_provider_model_id(llm_model_display_name)
            st.session_state.selected_llm_model_name = llm_model_name

            if llm_model_name == "PUT_PROVIDER_MODEL_ID_HERE":
                raise ValueError(
                    "The selected model still uses a placeholder provider model ID. "
                    "Update src/config/model_config.py with the real model ID before "
                    "running LLM generation."
                )

            st.session_state.metrics = calculate_context_metrics(
                full_text=st.session_state.document_text,
                naive_context=st.session_state.naive_context,
                engineered_context=st.session_state.engineered_context,
                model_name=llm_model_name,
            )

            naive_llm_start = time.perf_counter()
            st.session_state.naive_answer = generate_answer(
                question=question,
                context=st.session_state.naive_context,
                mode="Naive RAG",
                api_key=llm_api_key,
                base_url=llm_base_url,
                model_name=llm_model_name,
                temperature=llm_temperature,
            )
            st.session_state.processing_times["naive_llm_seconds"] = round(
                time.perf_counter() - naive_llm_start,
                4,
            )

            engineered_llm_start = time.perf_counter()
            st.session_state.engineered_answer = generate_answer(
                question=question,
                context=st.session_state.engineered_context,
                mode="Engineered Context Retrieval",
                api_key=llm_api_key,
                base_url=llm_base_url,
                model_name=llm_model_name,
                temperature=llm_temperature,
            )
            st.session_state.processing_times["engineered_llm_seconds"] = round(
                time.perf_counter() - engineered_llm_start,
                4,
            )

            st.session_state.naive_llm_token_estimate = calculate_llm_call_token_estimates(
                question=question,
                context=st.session_state.naive_context,
                answer=st.session_state.naive_answer,
                model_name=llm_model_name,
            )

            st.session_state.engineered_llm_token_estimate = (
                calculate_llm_call_token_estimates(
                    question=question,
                    context=st.session_state.engineered_context,
                    answer=st.session_state.engineered_answer,
                    model_name=llm_model_name,
                )
            )

            total_processing_seconds = round(
                time.perf_counter() - total_start_time,
                4,
            )
            st.session_state.processing_times["total_processing_seconds"] = (
                total_processing_seconds
            )

            log_process_event(
                build_analysis_log_event(
                    run_id=run_id,
                    event_type="analysis_run",
                    status="success",
                    total_processing_seconds=total_processing_seconds,
                    **log_settings,
                )
            )

            st.success(
                "Analysis complete: retrieval, re-ranking, contexts, metrics, and "
                "answers are ready. Review Tab 2 for ranking details and Tab 3 for "
                "comparison."
            )

        except Exception as run_error:
            total_processing_seconds = round(
                time.perf_counter() - total_start_time,
                4,
            )
            st.session_state.processing_times["total_processing_seconds"] = (
                total_processing_seconds
            )
            log_process_event(
                build_analysis_log_event(
                    run_id=run_id,
                    event_type="analysis_run",
                    status="failed",
                    error_message=str(run_error),
                    total_processing_seconds=total_processing_seconds,
                    **log_settings,
                )
            )
            st.error(f"Analysis run failed: {run_error}")

# Render metrics after the run block so the same rerun shows updated values.
st.subheader("Context Efficiency Metrics")
show_token_metric_disclaimer()
show_metric_cards(st.session_state.metrics)

# Three tabs for displaying different aspects of the context engineering process
tab_1, tab_2, tab_3 = st.tabs(
    [
        "Engineered Pipeline: Semantic Chunk Map",
        "Engineered Pipeline: Retrieval and Re-ranking Analytics",
        "Naive RAG vs Engineered Context",
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
    st.header("Engineered Pipeline: Semantic Chunk Map")

    st.write(
        "This tab shows the engineered pipeline preparation stages before final "
        "retrieval: sentence splitting, semantic distance, breakpoints, semantic "
        "chunks, metadata anchoring, and parent-child structure."
    )

    col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)

    with col_a:
        st.metric("Total Sentences", len(st.session_state.sentences))

    with col_b:
        st.metric("Naive Chunk Count", len(st.session_state.naive_chunks))

    with col_c:
        st.metric("Semantic Chunk Count", len(st.session_state.semantic_chunks))

    with col_d:
        st.metric("Breakpoints", len(st.session_state.semantic_breakpoints))

    with col_e:
        st.metric("Parent Chunk Count", len(st.session_state.parent_chunks))

    with col_f:
        st.metric("Child Chunk Count", len(st.session_state.child_chunks))

    st.divider()
    st.subheader("Naive RAG Pipeline Reference: Fixed-Size Naive Chunks")
    if st.session_state.naive_chunks:
        st.caption(
            "Baseline fixed-size chunks from the naive RAG pipeline. Compare these "
            "with engineered semantic chunks below."
        )
        with st.expander("View all naive chunks"):
            for chunk in st.session_state.naive_chunks:
                show_chunk_card(
                    title=f"Chunk {chunk['chunk_id']}",
                    caption=(
                        f"Characters {chunk['start_character']} to {chunk['end_character']} "
                        f"({chunk['character_count']} chars)"
                    ),
                    text=chunk["text"],
                )
    else:
        st.info("Naive chunks appear after you upload a document.")

    st.divider()
    st.subheader("1. Sentence Splitting Stage")
    show_formula_hint("Document → individual sentences")
    if st.session_state.sentences:
        show_formula_hint(
            "Each row is one sentence extracted from the uploaded document."
        )
        sentence_preview = st.session_state.sentences[:20]
        st.dataframe(
            sentence_preview,
            width="stretch",
            hide_index=True,
        )

        with st.expander("View all extracted sentences"):
            st.dataframe(
                st.session_state.sentences,
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("No sentences available.")
        st.info("Upload a document to generate sentence-level units.")

    st.divider()
    st.subheader("2. Semantic Distance Stage")
    show_formula_hint("Sentence embeddings → pairwise distance calculation")
    if st.session_state.sentence_distances:
        show_formula_hint(
            "Each row represents the semantic distance between neighboring sentences."
        )
        distance_df = pd.DataFrame(st.session_state.sentence_distances)
        st.dataframe(
            distance_df,
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Semantic Distance Spike Chart")
        char_df = distance_df[[
            "sentence_id_current",
            "cosine_distance"
        ]].set_index("sentence_id_current")
        st.line_chart(char_df)
        show_formula_hint(
            "Higher spikes indicate stronger semantic shifts between neighboring sentences."
        )

        st.subheader("Chunk Similarity Heatmap")
        if (
            st.session_state.chunk_similarity_matrix is not None
            and not st.session_state.chunk_similarity_matrix.empty
        ):
            try:
                styled_matrix = st.session_state.chunk_similarity_matrix.style.background_gradient(
                    cmap="Blues"
                )
                st.dataframe(styled_matrix, width="stretch")
            except Exception:
                st.dataframe(
                    st.session_state.chunk_similarity_matrix,
                    width="stretch",
                )
            show_formula_hint(
                "This heatmap shows how similar semantic chunks are to each other. "
                "Higher values mean the chunks discuss similar topics."
            )

        else:
            st.info(
                "Chunk similarity visuals appear after semantic chunks are created on upload."
            )
    else:
        st.info(
            "Upload a document to calculate sentence embeddings and semantic distances."
        )

    st.divider()
    st.subheader("3. Semantic Breakpoint Stage")
    show_formula_hint(
        "Distances above the selected threshold become chunk boundaries."
    )
    if st.session_state.semantic_breakpoints:
        st.write(f"Detected {len(st.session_state.semantic_breakpoints)} semantic breakpoints.")
        st.code(st.session_state.semantic_breakpoints)
    else:
        st.info("No semantic breakpoints detected.")
        st.info("Upload a document to calculate sentence embeddings and semantic distances.")

    st.divider()
    st.subheader("4. Semantic Chunk Stage")
    show_formula_hint("Distance spikes → semantic chunk boundaries")
    if st.session_state.semantic_chunks:
        show_formula_hint(
            "Semantic chunks group consecutive sentences between detected breakpoints."
        )
        semantic_chunk_table = []
        for chunk in st.session_state.semantic_chunks:
            semantic_chunk_table.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "sentence_start": chunk["sentence_start"],
                    "sentence_end": chunk["sentence_end"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "text_preview": chunk["text"][:200],
                }
            )

        st.dataframe(
            semantic_chunk_table,
            width="stretch",
            hide_index=True,
        )

        with st.expander("View all semantic chunks"):
            for chunk in st.session_state.semantic_chunks:
                show_chunk_card(
                    title=f"Chunk {chunk['chunk_id']}",
                    caption=(
                        f"Sentence {chunk['sentence_start']} to {chunk['sentence_end']}"
                        f" — Characters {chunk['char_start']} to {chunk['char_end']}"
                    ),
                    text=chunk["text"],
                )

    else:
        st.info("Semantic chunks will appear here after document processing.")

    st.divider()
    st.subheader("5. Metadata Anchoring Stage")
    show_formula_hint("Adds document metadata to improve retrieval quality")
    if st.session_state.document_metadata:
        st.json(st.session_state.document_metadata)
        if st.session_state.semantic_chunks:
            sample_chunk = st.session_state.semantic_chunks[0]
            with st.expander("View sample Metadata Anchored Embeddding Text"):
                st.text(build_embedding_texts(sample_chunk))
        elif st.session_state.naive_chunks:
            sample_chunk = st.session_state.naive_chunks[0]
            with st.expander("View sample Metadata Anchored Embeddding Text"):
                st.text(build_embedding_texts(sample_chunk))
        else:
            st.info("No chunks available to preview metadata anchoring.")
            st.info("Upload a document to generate chunks and metadata.")

    st.divider()
    st.subheader("6. Parent-Child Retrieval Stage")
    show_formula_hint(
        "Large parent chunks for context, smaller child chunks for retrieval"
    )

    if st.session_state.parent_chunks and st.session_state.child_chunks:
        show_formula_hint(
            "Parent chunks preserve broader context; child chunks are embedded for search."
        )
        parent_chunk_table = []
        for chunk in st.session_state.parent_chunks:
            related_children = [
                child for child in st.session_state.child_chunks if child["parent_id"] == chunk["parent_id"]
            ]
            parent_chunk_table.append(
                {
                    "parent_id": chunk["parent_id"],
                    "source_semantic_chunk_id": chunk["source_semantic_chunk_id"],
                    "sentence_start": chunk["sentence_start"],
                    "sentence_end": chunk["sentence_end"],
                    "child_count": len(related_children),
                    "parent_preview": chunk["parent_text"][:180],
                }
            )
        st.dataframe(
            parent_chunk_table, 
            width="stretch", 
            hide_index=True)
        
        with st.expander("View all parent chunks"):
            for chunk in st.session_state.parent_chunks:
                show_chunk_card(
                    title=f"Parent Chunk {chunk['parent_id']}",
                    caption=(
                        f"From Semantic Chunk {chunk['source_semantic_chunk_id']} — "
                        f"Sentence {chunk['sentence_start']} to {chunk['sentence_end']}"
                    ),
                    text=chunk["parent_text"],
                )

        child_chunk_table = []
        for chunk in st.session_state.child_chunks:
            child_chunk_table.append(
                {
                    "child_id": chunk["child_id"],
                    "parent_id": chunk["parent_id"],
                    "word_start": chunk["word_start"],
                    "word_end": chunk["word_end"],
                    "source_semantic_chunk_id": chunk["source_semantic_chunk_id"],
                    "child_preview": chunk["child_text"][:180],
                }
            )

        st.subheader("Child Chunk Preview")
        show_formula_hint(
            "Child chunks are the retrieval units indexed for vector similarity search."
        )
        st.dataframe(
            child_chunk_table,
            width="stretch",
            hide_index=True,
        )
        with st.expander("View Sample Child Embedding Text"):
            sample_chunk = st.session_state.child_chunks[0]
            st.text(build_child_embedding_text(sample_chunk))

        with st.expander("View all child chunks"):
            for chunk in st.session_state.child_chunks:
                show_chunk_card(
                    title=f"Child Chunk {chunk['child_id']}",
                    caption=(
                        f"From Parent Chunk {chunk['parent_id']} — "
                        f"Words {chunk['word_start']} to {chunk['word_end']}"
                    ),
                    text=chunk["child_text"],
                )
    else:
        st.info("Parent-child chunks will appear here after semantic chunks are processed.")


# ============================================================
# TAB 2: RE-RANKER ANALYTICS
# ============================================================
# This tab will later show:
# - top 10 raw vector search hits
# - vector similarity scores
# - cross-encoder re-ranker scores
# - final top 3 selected chunks

with tab_2:
    st.header("Engineered Pipeline: Retrieval and Re-ranking Analytics")

    show_info_box(
        "What this tab shows",
        "This tab shows how the engineered pipeline retrieves candidate child chunks "
        "and re-ranks them before final context construction.",
    )

    st.divider()

    st.subheader("Candidate Retrieval Stage: Raw Vector Search Results")
    show_formula_hint("Embedding similarity search over child chunks")

    if st.session_state.raw_vector_results:
        show_formula_hint(
            "Top candidate chunks returned by vector similarity search before re-ranking."
        )
        raw_results_table = []

        for result in st.session_state.raw_vector_results:
            raw_results_table.append(
                {
                    "raw_rank": result["raw_rank"],
                    "child_id": result["child_id"],
                    "parent_id": result["parent_id"],
                    "semantic_chunk": result["source_semantic_chunk_id"],
                    "vector_similarity": result["vector_similarity"],
                    "chunk_preview": result["chunk_preview"],
                }
            )

        st.dataframe(
            raw_results_table,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Raw Vector Hit Details")

        for result in st.session_state.raw_vector_results:
            with st.expander(
                f"Raw Rank {result['raw_rank']} | "
                f"Similarity {result['vector_similarity']} | "
                f"{result['child_id']}"
            ):
                st.markdown("#### Searchable Child Text")
                st.write(result["child_text"])

                st.markdown("#### Parent Text Kept for Later Context")
                st.write(result["parent_text"])

                st.markdown("#### Metadata")
                st.json(
                    {
                        "file_name": result["file_name"],
                        "parent_id": result["parent_id"],
                        "child_id": result["child_id"],
                        "source_semantic_chunk_id": result["source_semantic_chunk_id"],
                        "source_type": result["source_type"],
                    }
                )

        st.divider()

        st.subheader("Precision Ranking Stage: Cross-Encoder Re-ranked Results")
        show_formula_hint("Cross-encoder re-ranking of retrieved candidates")
        if st.session_state.reranked_results:
            show_formula_hint(
                "Same candidate chunks re-scored using a cross-encoder that evaluates "
                "the question and chunk together."
            )
            reranked_table = []

            for result in st.session_state.reranked_results:
                reranked_table.append(
                    {
                        "rerank_position": result["rerank_position"],
                        "raw_rank": result["raw_rank"],
                        "child_id": result["child_id"],
                        "parent_id": result["parent_id"],
                        "vector_similarity": result["vector_similarity"],
                        "reranker_score": result["reranker_score"],
                        "selected_for_final_context": result["selected"],
                        "chunk_preview": result["chunk_preview"],
                    }
                )

            st.dataframe(
                reranked_table,
                use_container_width=True,
                hide_index=True
            )

            show_formula_hint(
                "Re-ranked order may differ from raw vector order when deeper relevance "
                "scoring changes the ranking."
            )

    else:
        st.info(
            "Re-ranked results will appear here after running the analysis."
        )

    st.divider()

    st.subheader("Final Evidence Selection Stage")
    show_formula_hint("Selected evidence assembled into final LLM context")

    selected_results = get_selected_reranked_results(
        st.session_state.reranked_results
    )

    if selected_results:
        show_formula_hint(
            "Chunks selected for final engineered context construction."
        )
        selected_table = []

        for result in selected_results:
            selected_table.append(
                {
                    "rerank_position": result["rerank_position"],
                    "raw_rank": result["raw_rank"],
                    "child_id": result["child_id"],
                    "parent_id": result["parent_id"],
                    "reranker_score": result["reranker_score"],
                    "child_preview": result["child_text"][:250],
                }
            )

        st.dataframe(
            selected_table,
            use_container_width=True,
            hide_index=True
        )

        with st.expander("View Selected Re-ranked Chunk Details"):
            for result in selected_results:
                st.markdown(
                    f"### Selected Chunk {result['rerank_position']}"
                )

                st.caption(
                    f"Raw Rank: {result['raw_rank']} | "
                    f"Reranker Score: {result['reranker_score']} | "
                    f"Child ID: {result['child_id']} | "
                    f"Parent ID: {result['parent_id']}"
                )

                st.markdown("#### Child Text Used for Re-ranking")
                st.write(result["child_text"])

                st.markdown("#### Parent Text Kept for Final Context")
                st.write(result["parent_text"])

                st.divider()

    else:
        st.info(
            "Final selected chunks will appear here after re-ranking."
        )

    st.divider()

    st.subheader("Ranking Movement Analysis")

    if st.session_state.reranked_results:
        show_formula_hint(
            "Shows how cross-encoder re-ranking changed the order of retrieved chunks."
        )
        rank_comparison_table = []

        for result in st.session_state.reranked_results:
            rank_comparison_table.append(
                {
                    "child_id": result["child_id"],
                    "raw_rank": result["raw_rank"],
                    "rerank_position": result["rerank_position"],
                    "rank_change": result["raw_rank"] - result["rerank_position"],
                    "vector_similarity": result["vector_similarity"],
                    "reranker_score": result["reranker_score"],
                    "selected": result["selected"],
                }
            )

        st.dataframe(
            rank_comparison_table,
            use_container_width=True,
            hide_index=True
        )

        show_formula_hint(
            "Positive rank_change means the chunk moved higher after re-ranking; "
            "negative means it moved lower."
        )

    else:
        st.info(
            "Rank comparison will appear after re-ranking."
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

    st.write(
        "Compare final answers, contexts, approximate token metrics, and a stage-by-stage "
        "review of both pipelines."
    )

    st.divider()

    st.subheader("1. Final Answer Comparison")
    show_formula_hint(
        "Each pipeline sends its own retrieved context to the LLM with the same question."
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Naive RAG Answer")

        if st.session_state.naive_answer:
            st.write(st.session_state.naive_answer)
        else:
            st.info(
                "Naive answer will appear after running the full analysis."
            )

    with right_col:
        st.markdown("#### Engineered Context Answer")

        if st.session_state.engineered_answer:
            st.write(st.session_state.engineered_answer)
        else:
            st.info(
                "Engineered answer will appear after running the full analysis."
            )

    st.divider()

    st.subheader("2. Context Used for Each Pipeline")
    show_formula_hint(
        "Naive context uses raw vector hits; engineered context uses re-ranked parent text."
    )

    context_col_1, context_col_2 = st.columns(2)

    with context_col_1:
        show_context_with_chunk_expanders(
            "Naive RAG Context", st.session_state.naive_context
        )

    with context_col_2:
        show_context_with_chunk_expanders(
            "Engineered Context", st.session_state.engineered_context
        )

    st.divider()

    st.subheader("3. Context Efficiency Metrics")
    show_token_metric_disclaimer()
    show_context_difference_metrics(st.session_state.metrics)
    show_metric_cards(st.session_state.metrics)

    st.divider()

    st.subheader("4. Estimated LLM API Token Usage")
    show_token_metric_disclaimer()

    naive_est = st.session_state.naive_llm_token_estimate or {}
    engineered_est = st.session_state.engineered_llm_token_estimate or {}

    llm_col_1, llm_col_2 = st.columns(2)

    with llm_col_1:
        show_llm_token_estimate_column("Naive RAG LLM Call Estimate", naive_est)

    with llm_col_2:
        show_llm_token_estimate_column(
            "Engineered Context LLM Call Estimate", engineered_est
        )

    st.markdown("#### Combined Estimated API Usage")
    show_formula_hint(
        "Sum of naive and engineered LLM calls (two separate API requests per run)."
    )

    combined_input = (
        naive_est.get("estimated_input_tokens", 0)
        + engineered_est.get("estimated_input_tokens", 0)
    )
    combined_output = (
        naive_est.get("estimated_output_tokens", 0)
        + engineered_est.get("estimated_output_tokens", 0)
    )
    combined_total = (
        naive_est.get("estimated_total_tokens", 0)
        + engineered_est.get("estimated_total_tokens", 0)
    )

    combined_col_1, combined_col_2, combined_col_3 = st.columns(3)

    with combined_col_1:
        st.metric("Approx. Combined Estimated Input Tokens", combined_input)
        show_formula_hint("(Naive Input Tokens + Engineered Input Tokens)")

    with combined_col_2:
        st.metric("Approx. Combined Estimated Output Tokens", combined_output)
        show_formula_hint("(Naive Output Tokens + Engineered Output Tokens)")

    with combined_col_3:
        st.metric("Approx. Combined Estimated Total Tokens", combined_total)
        show_formula_hint("(Combined Input Tokens + Combined Output Tokens)")

    st.warning(
        "These token values use tiktoken (with cl100k_base fallback when needed). The "
        "provider dashboard may still show different numbers because it uses the "
        "model's real tokenizer and may include system messages, prompt formatting, "
        "context text, user question, generated answer, and provider-side overhead. "
        "The provider dashboard is the source of truth for actual API usage or billing."
    )
    
    #with st.expander("Why provider token usage may be higher"):
    #    st.markdown(
    #        """
#- **This app** uses tiktoken for counting, falling back to `cl100k_base` when the selected model is unknown.
#- **Your LLM provider** uses the actual tokenizer for the selected model, so counts often differ.
#- **Two API calls** are made: one for the naive RAG answer and one for the engineered-context answer. Provider usage usually includes both.
#- **Each call sends** the user question, retrieved context, a system/instruction message (we estimate ~100 tokens), and the model's generated answer.
#- **Provider dashboards** typically report **input tokens** (prompt + context + system) and **output tokens** (the answer) separately, then sum them for billing.
#- **Long retrieved contexts** can make provider input usage much larger than answer-only counts shown in the UI.
#        """
#        )

    st.divider()

    show_processing_time_summary(st.session_state.processing_times)

    st.divider()

    st.subheader("5. Pipeline Stage Review")

    token_model_name = (
        st.session_state.selected_llm_model_name
        or get_provider_model_id(llm_model_display_name)
    )
    naive_answer_tokens = count_tokens(
        st.session_state.naive_answer, token_model_name
    )
    engineered_answer_tokens = count_tokens(
        st.session_state.engineered_answer, token_model_name
    )

    selected_count = len(
        [
            result
            for result in st.session_state.reranked_results
            if result.get("selected") is True
        ]
    )

    review_col_1, review_col_2 = st.columns(2)

    with review_col_1:
        st.markdown("#### Naive RAG Pipeline")
        show_formula_hint("Fixed-size chunks → vector retrieval → direct context")
        st.metric("Fixed-Size Chunks Count", len(st.session_state.naive_chunks))
        st.metric("Raw Vector Retrieval Results", len(st.session_state.raw_vector_results))
        st.metric(
            "Tokenized Naive Context Tokens",
            st.session_state.metrics["naive_context_tokens"],
        )
        st.metric("Tokenized Naive Answer Tokens", naive_answer_tokens)

    with review_col_2:
        st.markdown("#### Engineered Context Pipeline")
        show_formula_hint(
            "Semantic chunking → metadata → parent-child → re-rank → engineered context"
        )
        st.metric("Sentence Count", len(st.session_state.sentences))
        st.metric("Semantic Chunk Count", len(st.session_state.semantic_chunks))
        st.metric("Breakpoint Count", len(st.session_state.semantic_breakpoints))
        st.metric("Parent Chunk Count", len(st.session_state.parent_chunks))
        st.metric("Child Chunk Count", len(st.session_state.child_chunks))
        st.metric("Re-ranked Candidate Count", len(st.session_state.reranked_results))
        st.metric("Selected Evidence Count", selected_count)
        st.metric(
            "Tokenized Engineered Context Tokens",
            st.session_state.metrics["engineered_context_tokens"],
        )
        st.metric("Tokenized Engineered Answer Tokens", engineered_answer_tokens)

    st.divider()

    show_pipeline_defaults_expander(get_pipeline_defaults())