"""
Reusable Streamlit UI helpers for the context engineering harness.

Keeps display logic separate from retrieval, chunking, and LLM pipeline code.
"""

import streamlit as st


def show_pipeline_summary():
    """
    Show the high-level two-pipeline comparison.
    """

    st.markdown("### Pipeline Summary")

    col_1, col_2 = st.columns(2)

    with col_1:
        st.markdown("#### Naive RAG Pipeline")
        st.info(
            "Fixed-size chunks → vector retrieval → direct context building → "
            "answer generation"
        )

    with col_2:
        st.markdown("#### Engineered Context Pipeline")
        st.success(
            "Sentence splitting → semantic chunking → metadata anchoring → "
            "parent-child retrieval → vector retrieval → cross-encoder re-ranking → "
            "engineered context → answer generation"
        )

    st.divider()


def show_metric_cards(metrics):
    """
    Show approximate token-efficiency metric cards.
    """

    metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)

    with metric_col_1:
        st.metric(
            label="Approx. Full Document Tokens",
            value=metrics.get("full_document_tokens", 0),
        )

    with metric_col_2:
        st.metric(
            label="Approx. Naive Context Tokens",
            value=metrics.get("naive_context_tokens", 0),
        )

    with metric_col_3:
        st.metric(
            label="Approx. Engineered Context Tokens",
            value=metrics.get("engineered_context_tokens", 0),
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
    Reusable chunk display card.
    """

    st.markdown(f"### {title}")
    st.caption(caption)
    st.write(text)
    st.divider()


def show_info_box(title, body):
    """
    Reusable explanation box.
    """

    st.markdown(f"#### {title}")
    st.info(body)


def show_warning_for_large_document(document_text):
    """
    Warn the user if the uploaded document is very large.
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
    Add a final interview guide section.
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


def show_approx_token_disclaimer():
    """
    Shared disclaimer for approximate token counts.
    """

    st.caption(
        "Token counts shown in this app are approximate. The provider dashboard is "
        "the source of truth because it uses the selected model's actual tokenizer "
        "and may include provider-side formatting overhead."
    )
