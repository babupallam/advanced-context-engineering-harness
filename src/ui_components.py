"""
Reusable Streamlit UI helpers for the context engineering harness.

Keeps display logic separate from retrieval, chunking, and LLM pipeline code.
"""

import streamlit as st


def show_pipeline_summary():
    """
    STEP 1:
    Show the high-level pipeline comparison.

    Why:
    This helps interviewers quickly understand what your app is comparing.
    """

    st.markdown("### Pipeline Summary")

    col_1, col_2 = st.columns(2)

    with col_1:
        st.markdown("#### Naive RAG")
        st.info(
            "Fixed chunks → vector search → raw top chunks → LLM answer"
        )

    with col_2:
        st.markdown("#### Engineered Context")
        st.success(
            "Semantic chunks → metadata anchoring → parent-child retrieval → "
            "vector search → cross-encoder re-ranking → compact context → LLM answer"
        )

    st.divider()


def show_metric_cards(metrics):
    """
    STEP 1:
    Show token-efficiency metric cards.

    Why:
    Metrics make the project look measurable instead of just visual.
    """

    metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)

    with metric_col_1:
        st.metric(
            label="Full Document Tokens",
            value=metrics.get("full_document_tokens", 0)
        )

    with metric_col_2:
        st.metric(
            label="Naive Context Tokens",
            value=metrics.get("naive_context_tokens", 0)
        )

    with metric_col_3:
        st.metric(
            label="Engineered Context Tokens",
            value=metrics.get("engineered_context_tokens", 0)
        )

    with metric_col_4:
        st.metric(
            label="Tokens Saved Percent",
            value=metrics.get("tokens_saved_percent", 0),
        )

    with metric_col_5:
        st.metric(
            label="Context Reduction Percent",
            value=metrics.get("context_reduction_percent", 0),
        )


def show_chunk_card(title, caption, text):
    """
    STEP 1:
    Reusable chunk display card.

    Why:
    Chunks appear in many places:
    - naive chunks
    - semantic chunks
    - parent chunks
    - child chunks

    This function keeps their display consistent.
    """

    st.markdown(f"### {title}")
    st.caption(caption)
    st.write(text)
    st.divider()


def show_info_box(title, body):
    """
    STEP 1:
    Reusable explanation box.

    Why:
    This app is for learning and interviews, so short explanations help
    users understand each technique.
    """

    st.markdown(f"#### {title}")
    st.info(body)


def show_warning_for_large_document(document_text):
    """
    STEP 1:
    Warn the user if the uploaded document is very large.

    Why:
    Large documents may slow down embedding, re-ranking, and LLM calls.
    """

    word_count = len(document_text.split()) if document_text else 0

    if word_count > 20000:
        st.warning(
            "This document is quite large. Processing may be slower because "
            "sentence embeddings, vector search, re-ranking, and answer generation "
            "need more compute."
        )


def show_interview_talking_points():
    """
    STEP 1:
    Add a final interview guide section.

    Why:
    This turns the app from a technical demo into a portfolio project.
    """

    with st.expander("Interview Talking Points"):
        st.markdown(
            """
### How to explain this project

I built a context engineering harness that compares naive RAG against an engineered retrieval pipeline.

### Why naive RAG is weak

Naive RAG often splits documents using fixed chunk sizes. This can cut ideas in half and retrieve shallow matches.

### What my engineered pipeline adds

The engineered version uses sentence-level semantic distance, metadata anchoring, parent-child retrieval, vector search, cross-encoder re-ranking, and token-efficiency metrics.

### Why re-ranking matters

Vector search is fast but approximate. The cross-encoder reads the question and chunk together, so it can improve relevance before the final context reaches the LLM.

### Business value

The system can reduce unnecessary context, improve answer grounding, and make retrieval behaviour easier to inspect.
"""
        )
