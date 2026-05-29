"""
Pipeline stage registry for the AI Retrieval & Context Engineering Workbench.

This module holds metadata only. It describes each stage for demos and interviews.
It does not run chunking, retrieval, or LLM logic.
"""

import pandas as pd


def get_pipeline_stages():
    """
    Return metadata for every workbench stage from naive RAG through observability.

    Each stage is a dictionary with fields used by the Pipeline Overview tab.
    """

    return [
        {
            "stage_id": "naive_rag",
            "stage_name": "Naive RAG",
            "short_name": "Naive RAG",
            "description": (
                "Baseline retrieval pipeline using fixed-size text chunks and "
                "direct vector similarity before any semantic or structural engineering."
            ),
            "techniques_used": [
                "fixed-size chunking",
                "raw retrieval baseline",
            ],
            "input_data": "Cleaned document text, chunk size, chunk overlap",
            "output_data": "naive_chunks, raw_vector_results, naive_context",
            "employer_skill_signal": (
                "Shows you can explain a simple RAG baseline and why stronger "
                "pipelines are needed for production quality."
            ),
        },
        {
            "stage_id": "semantic_chunking",
            "stage_name": "Semantic Chunking",
            "short_name": "Semantic Chunking",
            "description": (
                "Splits the document into sentences, measures semantic distance "
                "between neighbors, and groups sentences into meaning-based chunks."
            ),
            "techniques_used": [
                "sentence splitting",
                "sentence embeddings",
                "cosine distance",
                "semantic breakpoints",
            ],
            "input_data": "document_text, sentences, embedding model",
            "output_data": "sentences, sentence_distances, semantic_breakpoints, semantic_chunks",
            "employer_skill_signal": (
                "Demonstrates chunking beyond fixed windows and awareness of "
                "embedding-based document structure."
            ),
        },
        {
            "stage_id": "metadata_anchoring",
            "stage_name": "Metadata Anchoring",
            "short_name": "Metadata Anchoring",
            "description": (
                "Attaches document-level metadata to chunks so retrieval embeddings "
                "carry file and summary context, not only local chunk text."
            ),
            "techniques_used": [
                "document summary",
                "file name anchoring",
                "metadata-enriched embedding text",
            ],
            "input_data": "document_metadata, naive_chunks, semantic_chunks",
            "output_data": "Metadata-enriched chunk records ready for embedding",
            "employer_skill_signal": (
                "Shows practical context engineering: grounding chunks in source "
                "identity and document-level signals."
            ),
        },
        {
            "stage_id": "parent_child_retrieval",
            "stage_name": "Parent-Child Retrieval",
            "short_name": "Parent-Child Retrieval",
            "description": (
                "Builds larger parent chunks for readable context and smaller child "
                "chunks for precise search, so retrieval stays narrow but answers stay rich."
            ),
            "techniques_used": [
                "parent chunks",
                "child chunks",
                "search small answer with larger context",
            ],
            "input_data": "semantic_chunks, child_max_words, child_overlap_words",
            "output_data": "parent_chunks, child_chunks",
            "employer_skill_signal": (
                "Demonstrates hierarchical retrieval design used in production RAG "
                "systems that balance precision and context breadth."
            ),
        },
        {
            "stage_id": "vector_retrieval",
            "stage_name": "Vector Retrieval",
            "short_name": "Vector Retrieval",
            "description": (
                "Embeds the user question and child chunks, then retrieves the top-k "
                "candidates by cosine similarity from the vector index."
            ),
            "techniques_used": [
                "query embedding",
                "cosine similarity",
                "top-k retrieval",
            ],
            "input_data": "question, child_chunks, vector_index, raw_top_k",
            "output_data": "raw_vector_results",
            "employer_skill_signal": (
                "Core retrieval literacy: embedding search, similarity scoring, and "
                "candidate recall before precision ranking."
            ),
        },
        {
            "stage_id": "cross_encoder_reranking",
            "stage_name": "Cross-Encoder Re-ranking",
            "short_name": "Re-ranking",
            "description": (
                "Re-scores vector hits with a cross-encoder that reads the question and "
                "each chunk together, then selects the final evidence set."
            ),
            "techniques_used": [
                "query chunk pair scoring",
                "rerank position",
                "selected final chunks",
            ],
            "input_data": "question, raw_vector_results, reranker model, final_top_n",
            "output_data": "reranked_results",
            "employer_skill_signal": (
                "Shows two-stage retrieval (bi-encoder recall + cross-encoder precision) "
                "and how ranking changes context quality."
            ),
        },
        {
            "stage_id": "context_builder",
            "stage_name": "Context Builder",
            "short_name": "Context Builder",
            "description": (
                "Assembles the text blocks sent to the LLM: naive context from raw hits "
                "and engineered context from re-ranked parent text with deduplication."
            ),
            "techniques_used": [
                "naive context",
                "engineered context",
                "duplicate parent removal",
                "token efficiency",
            ],
            "input_data": "raw_vector_results, reranked_results, context chunk limits",
            "output_data": "naive_context, engineered_context, metrics",
            "employer_skill_signal": (
                "Highlights context window discipline, deduplication, and measurable "
                "token savings between pipeline designs."
            ),
        },
        {
            "stage_id": "llm_answer_generation",
            "stage_name": "LLM Answer Generation",
            "short_name": "LLM Answer",
            "description": (
                "Calls a general API provider with the same question and separate contexts "
                "to compare naive RAG answers against engineered-context answers."
            ),
            "techniques_used": [
                "grounded prompt",
                "general API provider",
                "naive vs engineered answer",
            ],
            "input_data": "question, naive_context, engineered_context, LLM secrets",
            "output_data": "naive_answer, engineered_answer, naive_llm_token_estimate, engineered_llm_token_estimate",
            "employer_skill_signal": (
                "End-to-end RAG delivery: grounded generation, provider integration, and "
                "side-by-side quality comparison."
            ),
        },
        {
            "stage_id": "prompt_engineering_lab",
            "stage_name": "Prompt Engineering Lab",
            "short_name": "Prompt Lab",
            "description": (
                "Future stage for experimenting with prompt templates, instructions, and "
                "refinement loops without changing the retrieval backend."
            ),
            "techniques_used": [
                "basic prompt",
                "structured prompt",
                "evidence-based prompt",
                "self-refinement prompt",
            ],
            "input_data": "question, retrieved context, prompt template variants",
            "output_data": "Prompt variants and comparative LLM outputs (planned)",
            "employer_skill_signal": (
                "Signals prompt ops maturity: systematic template design, evidence "
                "constraints, and iterative refinement."
            ),
        },
        {
            "stage_id": "evaluation_lab",
            "stage_name": "Evaluation Lab",
            "short_name": "Evaluation",
            "description": (
                "Future stage for scoring retrieval quality, answer groundedness, and "
                "hallucination risk against benchmarks or rubrics."
            ),
            "techniques_used": [
                "relevance scoring",
                "groundedness scoring",
                "hallucination risk",
                "completeness scoring",
            ],
            "input_data": "question, contexts, answers, golden references (planned)",
            "output_data": "Evaluation scores and run reports (planned)",
            "employer_skill_signal": (
                "Shows you think about measurable RAG quality, not only demo answers."
            ),
        },
        {
            "stage_id": "observability",
            "stage_name": "Observability",
            "short_name": "Observability",
            "description": (
                "Tracks how long each pipeline step takes, how many tokens are used, and "
                "how runs are logged for debugging and cost awareness."
            ),
            "techniques_used": [
                "latency tracking",
                "token usage tracking",
                "execution trace",
                "cost estimation",
            ],
            "input_data": "processing_times, metrics, process_logs.csv events",
            "output_data": "Timing summaries, token cards, historical run logs",
            "employer_skill_signal": (
                "Production mindset: observability, cost control, and traceable "
                "experiments for iterative pipeline improvement."
            ),
        },
    ]


def get_stage_by_id(stage_id):
    """
    Look up one stage dictionary by its stage_id string.

    Returns None if the id is not found.
    """

    for stage in get_pipeline_stages():
        if stage["stage_id"] == stage_id:
            return stage

    return None


def get_stage_names():
    """
    Return a simple list of stage_name values in pipeline order.
    """

    return [stage["stage_name"] for stage in get_pipeline_stages()]


def get_stage_summary_table():
    """
    Build a compact pandas table for the Pipeline Overview tab.

    Columns are chosen for quick scanning during demos and interviews.
    """

    rows = []

    for index, stage in enumerate(get_pipeline_stages(), start=1):
        rows.append(
            {
                "Step": index,
                "Stage": stage["stage_name"],
                "Short Name": stage["short_name"],
                "Key Techniques": ", ".join(stage["techniques_used"]),
                "Primary Output": stage["output_data"],
            }
        )

    return pd.DataFrame(rows)
