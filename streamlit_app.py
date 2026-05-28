import logging
import os
import warnings

# Suppress Hugging Face advisory noise when Streamlit inspects lazy transformers submodules.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"Accessing `__path__` from")

import pandas as pd
import streamlit as st

from src.document_loader import load_uploaded_document
from src.text_cleaner import clean_text
from src.naive_chunker import naive_chunk_text
from src.embeddings import load_embedding_model, embed_texts
from src.semantic_chunker import (
    split_into_sentences,
    calculate_sentence_distances,
    find_semantic_breakpoints,
    build_semantic_chunks
)
from src.metadata_anchor import create_document_metadata, attach_metadata_to_chunks, build_embedding_texts
from src.parent_child import create_parent_child_chunks, build_child_embedding_text
from src.vector_store import build_vector_index, retrieve_top_k
from src.reranker import load_reranker_model, rerank_results, get_selected_reranked_results

from src.context_builder import build_naive_context, build_engineered_context
from src.metrics import calculate_context_metrics
from src.llm_client import generate_answer


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
            "percentile",
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

    st.divider()

    st.subheader("Parent-Child Settings")

    child_max_words = st.number_input(
        "Child chunk max words",
        min_value=80,
        max_value=500,
        value=220,
        step=20
    )

    child_overlap_words = st.number_input(
        "Child chunk overlap words",
        min_value=0,
        max_value=150,
        value=40,
        step=10
    )

    st.divider()

    st.subheader("Context Building Settings")

    naive_context_max_chunks = st.number_input(
        "Naive context max chunks",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

    engineered_context_max_chunks = st.number_input(
        "Engineered context max chunks",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )


    st.divider()

    st.subheader("LLM Provider Settings")

    default_base_url = st.secrets.get(
        "LLM_BASE_URL",
        ""
    )

    default_model_name = st.secrets.get(
        "LLM_MODEL_NAME",
        ""
    )

    default_api_key = st.secrets.get(
        "LLM_API_KEY",
        ""
    )

    llm_base_url = st.text_input(
        "LLM API base URL",
        value=default_base_url,
        placeholder="Example: https://api.openai.com/v1"
    )

    llm_model_name = st.text_input(
        "LLM model name",
        value=default_model_name,
        placeholder="Example: gpt-4o-mini or provider-model-name"
    )

    llm_api_key = st.text_input(
        "LLM API key",
        value=default_api_key,
        type="password"
    )

    llm_temperature = st.slider(
        "LLM temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )


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
        document_text = load_uploaded_document(uploaded_file)
        cleaned_text = clean_text(document_text)

        st.session_state.document_text = cleaned_text
        st.session_state.uploaded_file_name = uploaded_file.name

        st.session_state.document_metadata = create_document_metadata(
            uploaded_file.name, 
            st.session_state.document_text
        )

        #Naive chunking

        # Generate naive chunks immediately after text extraction and cleaning
        naive_chunks = naive_chunk_text(text = st.session_state.document_text,chunk_size= naive_chunk_size,overlap= naive_chunk_overlap)
        st.session_state.naive_chunks = attach_metadata_to_chunks(naive_chunks, st.session_state.document_metadata)

        #Semantic chunking

        st.session_state.sentences = split_into_sentences(st.session_state.document_text)

        if st.session_state.sentences:
            sentence_texts = [sentence["text"] for sentence in st.session_state.sentences]
            embedding_model = get_cached_embedding_model(embedding_model_name)
            st.session_state.sentence_embeddings = embed_texts(sentence_texts, embedding_model)
            st.session_state.sentence_distances = calculate_sentence_distances(st.session_state.sentences, st.session_state.sentence_embeddings)
            st.session_state.semantic_breakpoints = find_semantic_breakpoints(
                distances = st.session_state.sentence_distances, 
                mode = threshold_mode
            )
            
            semantic_chunks = build_semantic_chunks(
                sentences=st.session_state.sentences,
                breakpoints=st.session_state.semantic_breakpoints,
            )
            st.session_state.semantic_chunks = attach_metadata_to_chunks(semantic_chunks, st.session_state.document_metadata)

            parent_chunks, child_chunks = create_parent_child_chunks(
                semantic_chunks=st.session_state.semantic_chunks,
                child_max_words=child_max_words,
                child_overlap_words=child_overlap_words
            )
            st.session_state.parent_chunks = parent_chunks
            st.session_state.child_chunks = child_chunks


            if st.session_state.child_chunks:
                st.session_state.vector_index = build_vector_index(
                    child_chunks=st.session_state.child_chunks,
                    embedding_model=embedding_model
                )
            
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
        embedding_model = get_cached_embedding_model(
            embedding_model_name
        )

        st.session_state.raw_vector_results = retrieve_top_k(
            query=question,
            child_chunks=st.session_state.child_chunks,
            index=st.session_state.vector_index,
            embedding_model=embedding_model,
            top_k=raw_top_k
        )


        reranker_model = get_cached_reranker_model(
            reranker_model_name
        )

        st.session_state.reranked_results = rerank_results(
            query=question,
            raw_results=st.session_state.raw_vector_results,
            reranker_model=reranker_model,
            top_n=final_top_n
        )

        # build the naive context
        # This context is built from the raw vector results directly without re-ranking or duplicate parent removal
        st.session_state.naive_context = build_naive_context(
            raw_vector_results=st.session_state.raw_vector_results,
            max_chunks=naive_context_max_chunks
        )
        # build the engineered context
        # This context is built from the re-ranked results with duplicate parent removal
        st.session_state.engineered_context = build_engineered_context(
            reranked_results=st.session_state.reranked_results
        )

        # calculate the context metrics
        # These metrics help us understand how much context we are saving by using engineered context instead of naive context
        st.session_state.metrics = calculate_context_metrics(
            full_text=st.session_state.document_text,
            naive_context=st.session_state.naive_context,
            engineered_context=st.session_state.engineered_context
        )

        """
        Generate the naive answer.

        This answer uses naive_context.
        It is kept separate for comparison.
        """

        st.session_state.naive_answer = generate_answer(
            question=question,
            context=st.session_state.naive_context,
            mode="Naive RAG",
            api_key=llm_api_key,
            base_url=llm_base_url,
            model_name=llm_model_name,
            temperature=llm_temperature
        )

        """
        Generate the engineered context answer.

        This answer uses engineered_context.
        It is kept separate from the naive answer.
        """

        st.session_state.engineered_answer = generate_answer(
            question=question,
            context=st.session_state.engineered_context,
            mode="Engineered Context Retrieval",
            api_key=llm_api_key,
            base_url=llm_base_url,
            model_name=llm_model_name,
            temperature=llm_temperature
        )


        st.success(
            "Raw vector retrieval and cross-encoder re-ranking completed. "
            "Open Tab 2 to compare both ranking versions."
        )
               

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
    st.header("Sentence Splitting Preview")
    if st.session_state.sentences:
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
    st.subheader("Sentence - to - Sentence Cosine Distance Preview")
    if st.session_state.sentence_distances:
        distance_df = pd.DataFrame(st.session_state.sentence_distances)
        st.dataframe(distance_df,
        width="stretch",
        hide_index=True,
        )

        st.subheader("Semantic Distance Spike Chart") # plot the cosine distance values
        
        char_df = distance_df[[
            "sentence_id_current",
            "cosine_distance"
        ]].set_index("sentence_id_current")
        st.line_chart(char_df)

        st.info(
            "Higher spikes mean neighbouring sentences are less similar. "
            "In the next step, we will use these spikes as semantic breakpoints."
        )
    else:
        st.info(
            "Upload a document to calculate sentence embeddings and semantic distances."
        )

    st.divider()
    st.subheader("Detected Semantic Breakpoints")
    if st.session_state.semantic_breakpoints:
        st.write(f"Detected {len(st.session_state.semantic_breakpoints)} semantic breakpoints.")
        st.code(st.session_state.semantic_breakpoints)
    else:
        st.info("No semantic breakpoints detected.")
        st.info("Upload a document to calculate sentence embeddings and semantic distances.")
    
    st.divider()
    st.subheader("Metadata Anchoring Preview")
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
    st.subheader("Final Semantic Chunks")
    if st.session_state.semantic_chunks:
        #make it as a table
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
                st.markdown(f"**Chunk {chunk['chunk_id']}**")
                st.caption(
                    f"Sentence {chunk['sentence_start']} to {chunk['sentence_end']}"
                    f" - Characters {chunk['char_start']} to {chunk['char_end']}"
                )
                st.write(chunk['text'])
                 
                st.divider()

    else:
        st.info("Semantic chunks will appear here after document processing.")

    st.divider()
    st.subheader("Parent-Child Retrieval Version")
    st.write(
        "This section keeps semantic chunks as parent chunks and creates smaller "
        "child chunks for precise retrieval. The previous naive and semantic versions "
        "are still kept for comparison."
    )

    if st.session_state.parent_chunks and st.session_state.child_chunks:
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
                st.markdown(f"**Parent Chunk {chunk['parent_id']}**")
                st.caption(f"From Semantic Chunk {chunk['source_semantic_chunk_id']} - Sentence {chunk['sentence_start']} to {chunk['sentence_end']}")
                st.write(chunk['parent_text'])
                st.divider()

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
                st.markdown(f"**Child Chunk {chunk['child_id']}**")
                st.caption(f"From Parent Chunk {chunk['parent_id']} - Words {chunk['word_start']} to {chunk['word_end']}")
                st.write(chunk['child_text'])
                st.divider()
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
    st.header("Re-ranker Analytics")

    st.write(
        "This tab now shows raw vector retrieval results from the parent-child version. "
        "The app searches child chunks for precision while keeping parent text for later context."
    )

    st.divider()

    st.subheader("Version 4: Raw Vector Search Results")

    if st.session_state.raw_vector_results:
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

        st.subheader("Version 5: Cross-encoder Re-ranker Results")
        if st.session_state.reranked_results:
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

            st.info(
                "Notice that the re-ranked order may differ from the raw vector order. "
                "This is the point of re-ranking: it performs deeper relevance scoring."
            )

    else:
        st.info(
            "Re-ranked results will appear here after running the analysis."
        )

    st.divider()

    st.subheader("Final Selected Top Chunks")

    selected_results = get_selected_reranked_results(
        st.session_state.reranked_results
    )

    if selected_results:
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

    st.subheader("Raw Rank vs Re-ranked Position")

    if st.session_state.reranked_results:
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

        st.caption(
            "Positive rank_change means the chunk moved higher after re-ranking. "
            "Negative rank_change means it moved lower."
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
        "This tab compares the baseline naive context against the engineered context. "
        "All previous versions are kept available for review."
    )

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Naive RAG Answer")

        if st.session_state.naive_answer:
            st.write(st.session_state.naive_answer)

            st.metric(
                "Naive Context Tokens",
                st.session_state.metrics["naive_context_tokens"]
            )

            with st.expander("View Naive Context Used"):
                st.text_area(
                    "Naive Context",
                    value=st.session_state.naive_context,
                    height=400,
                    disabled=True
                )
        else:
            st.info(
                "Naive answer will appear after running the full analysis."
            )

    with right_col:
        st.subheader("Engineered Context Answer")

        if st.session_state.engineered_answer:
            st.write(st.session_state.engineered_answer)

            st.metric(
                "Engineered Context Tokens",
                st.session_state.metrics["engineered_context_tokens"]
            )

            with st.expander("View Engineered Context Used"):
                st.text_area(
                    "Engineered Context",
                    value=st.session_state.engineered_context,
                    height=400,
                    disabled=True
                )
        else:
            st.info(
                "Engineered answer will appear after running the full analysis."
            )

    st.divider()

    st.subheader("Token Efficiency Comparison")

    metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)

    with metric_a:
        st.metric(
            "Full Document Tokens",
            st.session_state.metrics["full_document_tokens"]
        )

    with metric_b:
        st.metric(
            "Naive Context Tokens",
            st.session_state.metrics["naive_context_tokens"]
        )

    with metric_c:
        st.metric(
            "Engineered Tokens",
            st.session_state.metrics["engineered_context_tokens"]
        )

    with metric_d:
        st.metric(
            "Tokens Saved %",
            f'{st.session_state.metrics["tokens_saved_percent"]}%'
        )

    with metric_e:
        st.metric(
            "Context Reduction %",
            f'{st.session_state.metrics["context_reduction_percent"]}%'
        )

    st.divider()

    with st.expander("Compare Available Pipeline Versions"):
        st.markdown("### Version 1: Naive Fixed-Size Chunks")

        st.write(
            "Baseline chunking version. It splits by character count and may cut ideas."
        )

        st.metric(
            "Naive Chunk Count",
            len(st.session_state.naive_chunks)
        )

        st.markdown("### Version 2: Semantic Chunks")

        st.write(
            "Meaning-based chunking version. It uses sentence distance spikes."
        )

        st.metric(
            "Semantic Chunk Count",
            len(st.session_state.semantic_chunks)
        )

        st.markdown("### Version 3: Parent-Child Retrieval Chunks")

        st.write(
            "Uses semantic chunks as larger parent context and smaller child chunks "
            "for precise search."
        )

        col_parent, col_child = st.columns(2)

        with col_parent:
            st.metric(
                "Parent Chunk Count",
                len(st.session_state.parent_chunks)
            )

        with col_child:
            st.metric(
                "Child Chunk Count",
                len(st.session_state.child_chunks)
            )

        st.markdown("### Version 4: Raw Vector Retrieval Results")

        st.write(
            "Uses the question to search child chunks by vector similarity."
        )

        st.metric(
            "Raw Vector Results",
            len(st.session_state.raw_vector_results)
        )

        st.markdown("### Version 5: Cross-Encoder Re-ranked Results")

        st.write(
            "Uses a cross-encoder to re-rank raw vector candidates more precisely."
        )

        selected_count = len(
            [
                result
                for result in st.session_state.reranked_results
                if result.get("selected") is True
            ]
        )

        col_rerank_all, col_rerank_selected = st.columns(2)

        with col_rerank_all:
            st.metric(
                "Re-ranked Candidates",
                len(st.session_state.reranked_results)
            )

        with col_rerank_selected:
            st.metric(
                "Selected Final Chunks",
                selected_count
            )

        st.markdown("### Version 6: Context Builder and Token Metrics")

        st.write(
            "Builds final naive and engineered contexts separately, then compares "
            "their approximate token usage."
        )

        col_naive_context, col_engineered_context = st.columns(2)

        with col_naive_context:
            st.metric(
                "Naive Context Tokens",
                st.session_state.metrics["naive_context_tokens"]
            )

        with col_engineered_context:
            st.metric(
                "Engineered Context Tokens",
                st.session_state.metrics["engineered_context_tokens"]
            )