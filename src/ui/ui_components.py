"""
Reusable Streamlit UI helpers for the context engineering harness.

Keeps display logic separate from retrieval, chunking, and LLM pipeline code.
"""

import pandas as pd
import streamlit as st


def show_formula_hint(text):
    """
    Display a compact explanation beneath metrics or values.
    """

    st.caption(text)


def _pipeline_status_label(status):
    if status == "must_replace":
        return "⚠️ must_replace"
    if status == "estimate":
        return "ℹ️ estimate"
    if status == "tunable_default":
        return "✅ tunable_default"
    if status == "configured":
        return "✅ configured"
    return status


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


def show_token_metric_disclaimer():
    """
    Shared disclaimer for tiktoken-based token counts.
    """

    st.caption(
        "Token counts use tiktoken where possible. For third-party models not known "
        "to tiktoken, the app falls back to cl100k_base, so provider dashboards may "
        "still differ."
    )


def show_context_change_help():
    """
    Explain signed context change metrics.
    """

    st.caption(
        "Positive means the engineered context is larger than the naive context. "
        "Negative means the engineered context is smaller."
    )


def show_metric_cards(metrics):
    """
    Compact token-efficiency metric cards (main summary row).
    """

    metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)

    with metric_col_1:
        st.metric(
            label="Approx. Full Document Tokens",
            value=metrics.get("full_document_tokens", 0),
        )
        show_formula_hint("(Tokenized full uploaded document using tiktoken.)")

    with metric_col_2:
        st.metric(
            label="Approx. Naive Context Tokens",
            value=metrics.get("naive_context_tokens", 0),
        )
        show_formula_hint("(Tokens sent to the LLM using the naive retrieval pipeline.)")

    with metric_col_3:
        st.metric(
            label="Approx. Engineered Context Tokens",
            value=metrics.get("engineered_context_tokens", 0),
        )
        show_formula_hint("(Tokens sent to the LLM using the engineered retrieval pipeline.)")

    with metric_col_4:
        st.metric(
            label="Context Change Percent",
            value=metrics.get("context_change_percent", 0),
        )
        show_formula_hint("((Engineered − Naive) / Naive) × 100")

    with metric_col_5:
        st.metric(
            label="Context Reduction Percent",
            value=metrics.get("context_reduction_percent", 0),
        )
        show_formula_hint("((Full Document − Engineered Context) / Full Document) × 100")


def show_context_difference_metrics(metrics):
    """
    Show context size difference and change percent with help text (Tab 3).
    """

    diff_col_1, diff_col_2 = st.columns(2)

    with diff_col_1:
        st.metric(
            label="Context Size Difference",
            value=metrics.get("context_size_difference", 0),
        )
        show_formula_hint("(Engineered Context Tokens − Naive Context Tokens)")

    with diff_col_2:
        st.metric(
            label="Context Change Percent",
            value=metrics.get("context_change_percent", 0),
        )
        show_formula_hint("((Engineered − Naive) / Naive) × 100")

    show_context_change_help()


def show_llm_token_estimate_column(title, estimate):
    """
    Display estimated LLM API token metrics with formula hints.
    """

    st.markdown(f"#### {title}")
    st.metric(
        "Approx. Estimated Input Tokens",
        estimate.get("estimated_input_tokens", 0),
    )
    show_formula_hint("(Question Tokens + Context Tokens + System Prompt Estimate)")

    st.metric(
        "Approx. Estimated Output Tokens",
        estimate.get("estimated_output_tokens", 0),
    )
    show_formula_hint("(Generated Answer Tokens)")

    st.metric(
        "Approx. Estimated Total Tokens",
        estimate.get("estimated_total_tokens", 0),
    )
    show_formula_hint("(Input Tokens + Output Tokens)")


def show_pipeline_defaults_expander(pipeline_defaults):
    """
    Display hardcoded or default pipeline configuration values.
    """

    with st.expander("Pipeline Defaults and Hardcoded Values"):
        rows = []
        for key, entry in pipeline_defaults.items():
            rows.append(
                {
                    "Parameter": key,
                    "Current Value": entry["value"],
                    "Used In": entry["used_in"],
                    "Reason": entry["reason"],
                    "Status": _pipeline_status_label(entry["status"]),
                }
            )

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        show_formula_hint(
            "Status: ✅ tunable_default / configured · ℹ️ estimate · ⚠️ must_replace"
        )

        must_replace_keys = [
            key
            for key, entry in pipeline_defaults.items()
            if entry["status"] == "must_replace"
        ]
        if must_replace_keys:
            st.warning(
                "Parameters marked ⚠️ must_replace should be updated before production: "
                + ", ".join(must_replace_keys)
            )

        estimate_keys = [
            key for key, entry in pipeline_defaults.items() if entry["status"] == "estimate"
        ]
        if estimate_keys:
            st.info(
                "Parameters marked ℹ️ estimate are approximations (not exact API counts): "
                + ", ".join(estimate_keys)
            )

        tunable_keys = [
            key
            for key, entry in pipeline_defaults.items()
            if entry["status"] == "tunable_default"
        ]
        if tunable_keys:
            st.success(
                "Parameters marked ✅ tunable_default can be adjusted in code or UI: "
                + ", ".join(tunable_keys)
            )


def context_chunk_label(chunk_text, index=0):
    """
    Derive a short expander title from a formatted context chunk block.
    """

    text = chunk_text.strip()
    if not text:
        return f"Chunk {index + 1}"

    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "Child ID":
            return value
        if key in ("Parent ID", "Parent ID Matched"):
            return f"{key}: {value}"
        if key == "Raw Rank":
            return f"Rank {value}"
        if key == "Rerank Position":
            return f"Rerank #{value}"

    first_line = text.splitlines()[0].strip().strip("[]")
    return first_line or f"Chunk {index + 1}"


def show_context_with_chunk_expanders(title, context_text):
    """
    Show assembled LLM context split into per-chunk expanders.
    """

    with st.expander(title):
        if not context_text or not context_text.strip():
            st.info("Context will appear after running the full analysis.")
            return

        chunks = [
            chunk.strip()
            for chunk in context_text.split("\n\n---\n\n")
            if chunk.strip()
        ]

        if not chunks:
            st.text(context_text)
            return

        for index, chunk in enumerate(chunks):
            label = context_chunk_label(chunk, index)
            with st.expander(f"Chunk {index + 1}: {label}"):
                st.text(chunk)


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
