Below is a **full learning-first roadmap** for your Streamlit portfolio project. The goal is not just to build another RAG app, but to build an **Advanced Context Engineering Harness** that clearly shows why engineered context is better than naive RAG.

Your uploaded document focuses on advanced context management: avoiding context rot, monitoring token usage, offloading heavy raw data, compacting before summarizing, using structured schemas, and keeping the LLM context clean rather than dumping everything into the prompt. Those ideas fit perfectly into this project because your app will visually prove how context engineering reduces token bloat and improves retrieval quality.  

---

# Project Title

## Advanced Context Engineering Harness

### Portfolio tagline

```text
A Streamlit application that compares naive RAG against engineered context retrieval using semantic chunking, metadata anchoring, parent-child retrieval, cross-encoder re-ranking, and token-efficiency metrics.
```

### How you should describe it in interviews

```text
I built a context engineering harness designed to solve the structural failures of naive RAG, especially token bloat, semantic fragmentation, and the lost-in-the-middle problem. Instead of fixed-size chunking, the app uses semantic breakpoints based on sentence-level embedding distance. It then applies parent-child retrieval and cross-encoder re-ranking to reduce the final LLM context while keeping the most relevant evidence.
```

---

# 1. What the Application Must Do

## User inputs

The Streamlit app should allow the user to upload:

```text
1. A long PDF document
2. A long TXT document
3. A complex user question
```

The complex question should be something like:

```text
What are the main risks of this system, and how does the author suggest reducing them?
```

or:

```text
Compare the technical recommendations in the document and explain which ones are most important for production deployment.
```

---

# 2. Final Streamlit Layout

The final app should have a clean structure like this:

```text
Advanced Context Engineering Harness

Sidebar:
- Upload PDF or TXT
- Enter complex question
- Select embedding model
- Select re-ranker model
- Set semantic threshold mode
- Set top-k vector retrieval count
- Set top-n reranked chunks
- Run Analysis button

Main Area:
- Metrics cards
- Three tabs

Tab 1: Semantic Chunk Map
Tab 2: Re-ranker Analytics
Tab 3: Naive RAG vs Engineered Context
```

---

# 3. Three Required Tabs

## Tab 1: Semantic Chunk Map

### Purpose

This tab should prove that your app is not blindly splitting text every 500 characters.

It should show how the document was split based on meaning shifts.

### What to display

```text
1. Total sentences extracted
2. Naive chunk count
3. Semantic chunk count
4. Average semantic chunk size
5. Sentence-to-sentence cosine distance table
6. Breakpoint positions
7. Visual chart showing semantic distance spikes
8. List of final semantic chunks
```

### What the interviewer should understand

Naive RAG uses fixed chunk sizes. Your app uses semantic shift detection.

The document you uploaded also emphasizes that large raw content should not remain inside the LLM context, and that systems should actively manage what enters the context window. 

### Core logic

```text
Step 1: Extract document text
Step 2: Split text into sentences
Step 3: Embed each sentence
Step 4: Compare each sentence with the next sentence
Step 5: Calculate cosine distance
Step 6: Find unusually high distance points
Step 7: Use those points as semantic breakpoints
Step 8: Build chunks around real meaning shifts
```

---

## Tab 2: Re-ranker Analytics

### Purpose

This tab should show the difference between fast vector search and deeper relevance scoring.

### What to display

```text
1. Top 10 raw vector search results
2. Similarity score for each raw hit
3. Chunk preview
4. Parent section ID
5. Metadata anchor
6. Cross-encoder re-rank score
7. Final Top 3 selected chunks
8. Difference between raw rank and re-ranked position
```

### Suggested table columns

```text
Raw Rank
Chunk ID
Parent ID
Vector Similarity
Re-ranker Score
Selected for Final Context
Chunk Preview
```

### What the interviewer should understand

Vector search is fast but not always precise. The re-ranker checks the query and chunk together, so it can remove weak matches before sending anything to the LLM.

This connects with the uploaded document’s broader rule: do not keep token-heavy raw content in the active context; keep only the most useful compact evidence. 

---

## Tab 3: Naive RAG vs Engineered Context

### Purpose

This is the most important portfolio tab.

It should show a side-by-side comparison:

```text
Left side: Naive RAG answer
Right side: Engineered Context answer
```

### Naive RAG pipeline

```text
PDF/TXT
→ fixed-size chunks
→ vector search
→ top chunks directly sent to LLM
→ answer
```

### Engineered context pipeline

```text
PDF/TXT
→ sentence splitting
→ semantic chunking
→ metadata anchoring
→ parent-child retrieval
→ vector search top 10
→ cross-encoder re-ranking
→ final top 3 parent contexts
→ compact prompt
→ answer
```

### What to display

```text
1. Naive answer
2. Engineered answer
3. Naive context chunks used
4. Engineered context chunks used
5. Token usage comparison
6. Context reduction percentage
7. Tokens saved percentage
8. Retrieval quality explanation
```

---

# 4. Metrics Cards

At the top of the app, show cards like:

```text
Full Document Tokens
Naive Context Tokens
Engineered Context Tokens
Tokens Saved %
Context Reduction vs Full Document %
Top-K Raw Hits
Final Re-ranked Chunks
```

### Important formulas

```text
Tokens Saved % = ((Naive Context Tokens - Engineered Context Tokens) / Naive Context Tokens) * 100
```

```text
Context Reduction vs Full Document % = ((Full Document Tokens - Engineered Context Tokens) / Full Document Tokens) * 100
```

### Example

```text
Full Document Tokens: 24,000
Naive Context Tokens: 7,500
Engineered Context Tokens: 3,000
Tokens Saved: 60%
Context Reduction vs Full Document: 87.5%
```

The uploaded document repeatedly emphasizes context monitoring, token thresholds, and reducing active context before degradation happens. That should become a visible engineering metric in your app, not just a hidden backend detail. 

---

# 5. Core Technical Concepts You Must Learn While Building

## Concept 1: Naive Chunking

You split text using fixed character or token length.

Example:

```text
Chunk 1: characters 0-1000
Chunk 2: characters 1001-2000
Chunk 3: characters 2001-3000
```

### Problem

It may cut an idea in half.

A sentence like this:

```text
The system failed because the memory layer retained raw documents in the active prompt...
```

may be separated from the explanation that follows.

---

## Concept 2: Semantic Chunking

You split where meaning changes.

### Learning objective

You should understand:

```text
Sentence embeddings
Cosine similarity
Cosine distance
Statistical threshold
Semantic breakpoint
```

### Simple logic

```text
Similarity high = same topic continues
Similarity low = topic changed
Distance spike = possible chunk boundary
```

---

## Concept 3: Metadata Anchoring

Each chunk should know where it came from.

### Example metadata

```text
Document title
Page number
Section heading
Parent chunk ID
Short document summary
Source file name
```

### Why this matters

A chunk like this is weak:

```text
Revenue increased by 12%.
```

A metadata-anchored chunk is stronger:

```text
Document: 2024 Annual Report
Section: Financial Performance
Page: 14

Revenue increased by 12%.
```

The uploaded document also recommends keeping identifiers and structured references instead of raw heavy data inside the active context. 

---

## Concept 4: Parent-Child Retrieval

This is a powerful interview point.

### Child chunk

Small chunk used for precise search.

```text
Around 150-250 tokens
```

### Parent chunk

Larger context passed to the LLM.

```text
Around 800-1200 tokens
```

### Why this works

```text
Search small.
Answer with bigger context.
```

This avoids both extremes:

```text
Too small = missing context
Too large = token waste
```

---

## Concept 5: Cross-Encoder Re-ranking

### Bi-encoder retrieval

Embeds query and chunks separately.

```text
Query → embedding
Chunk → embedding
Compare vectors
```

Fast, but sometimes shallow.

### Cross-encoder re-ranking

Reads query and chunk together.

```text
[Question + Chunk] → relevance score
```

Slower, but more accurate.

### Portfolio explanation

```text
I use vector search as a fast candidate generator, then use a cross-encoder as a precision layer.
```

---

# 6. Recommended Tech Stack

## For MVP

```text
Python
Streamlit
PyMuPDF or pypdf
sentence-transformers
FAISS
pandas
numpy
scikit-learn
tiktoken or simple tokenizer fallback
OpenAI / Gemini / Groq for final answer generation
```

## For deployment-friendly version

Use a smaller re-ranker first:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Later, for a more advanced local demo, you can experiment with heavier re-rankers.

Streamlit Community Cloud apps need declared Python dependencies, commonly through `requirements.txt`; the official docs state that missing dependency files are a common reason apps fail to build. ([Streamlit Docs][1])

---

# 7. Suggested Repository Structure

Use this structure because it teaches clean AI engineering.

```text
advanced-context-engineering-harness/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── document_loader.py
│   ├── text_cleaner.py
│   ├── naive_chunker.py
│   ├── semantic_chunker.py
│   ├── metadata_anchor.py
│   ├── parent_child.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── reranker.py
│   ├── context_builder.py
│   ├── llm_client.py
│   ├── metrics.py
│   └── ui_components.py
│
├── sample_docs/
│   └── sample_context_engineering.txt
│
└── .streamlit/
    └── config.toml
```

Do not commit secret files. Streamlit supports secrets through `st.secrets`, and the docs describe it as a dictionary-like interface for values stored in `secrets.toml`. ([Streamlit Docs][2])

---

# 8. GitHub Repository Creation

## Step 1: Create local folder

```bash
mkdir advanced-context-engineering-harness
cd advanced-context-engineering-harness
```

## Step 2: Create virtual environment

```bash
python -m venv venv
```

## Step 3: Activate virtual environment

```bash
venv\Scripts\activate
```

For Mac/Linux:

```bash
source venv/bin/activate
```

## Step 4: Create first files

```bash
mkdir src sample_docs .streamlit
type nul > streamlit_app.py
type nul > requirements.txt
type nul > README.md
type nul > .gitignore
type nul > .streamlit/config.toml
```

For Mac/Linux:

```bash
mkdir src sample_docs .streamlit
touch streamlit_app.py requirements.txt README.md .gitignore .streamlit/config.toml
```

## Step 5: Initialize Git

```bash
git init
git add .
git commit -m "Initial project structure"
```

## Step 6: Create GitHub repo

Create a new GitHub repo named:

```text
advanced-context-engineering-harness
```

Then connect it:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/advanced-context-engineering-harness.git
git push -u origin main
```

---

# 9. Development Phases With Codex Prompts

Use these prompts one by one in VS Code Codex. Do not ask Codex to build everything at once. Each phase should produce a working checkpoint.

---

## Phase 1: Streamlit UI Skeleton

### Goal

Build the visual structure only.

### What you learn

```text
Streamlit layout
Sidebar inputs
Tabs
Metrics cards
Session state
Clean UI planning
```

### Expected output

The app should run, but no AI logic yet.

### Codex prompt

```text
Create the initial Streamlit application for an app called "Advanced Context Engineering Harness".

Important:
- Do not build the full backend yet.
- Focus only on a clean UI skeleton.
- Use streamlit_app.py as the main file.
- Use comments to explain every major section.

Requirements:
1. Add a page title and tagline.
2. Add a sidebar with:
   - File uploader accepting PDF and TXT files.
   - Text area for a complex question.
   - Number input for raw vector top-k, default 10.
   - Number input for final re-ranked top-n, default 3.
   - Selectbox for semantic threshold mode: percentile, standard_deviation.
   - Button called "Run Context Engineering Analysis".
3. In the main area, create metric cards for:
   - Full Document Tokens
   - Naive Context Tokens
   - Engineered Context Tokens
   - Tokens Saved %
   - Context Reduction %
4. Create three tabs:
   - Semantic Chunk Map
   - Re-ranker Analytics
   - Naive RAG vs Engineered Context
5. Add placeholder text inside each tab explaining what will appear there.
6. Use st.session_state to store uploaded text, chunks, retrieval results, metrics, and answers.
7. Keep the code simple, readable, and beginner-friendly.
```

### Test

Run:

```bash
streamlit run streamlit_app.py
```

### Commit

```bash
git add .
git commit -m "Create Streamlit UI skeleton"
```

---

## Phase 2: Document Upload and Text Extraction

### Goal

Allow PDF/TXT upload and extract clean text.

### What you learn

```text
File handling
PDF parsing
Text cleaning
Streamlit upload state
Error handling
```

### Files to create

```text
src/document_loader.py
src/text_cleaner.py
```

### Codex prompt

```text
Add document loading functionality to the project.

Important:
- Do not change the UI design unnecessarily.
- Keep the app beginner-friendly.
- Add comments explaining each step.

Requirements:
1. Create src/document_loader.py.
2. Add functions for:
   - extract_text_from_pdf(uploaded_file)
   - extract_text_from_txt(uploaded_file)
   - load_uploaded_document(uploaded_file)
3. Use PyMuPDF if available for PDF extraction.
4. Create src/text_cleaner.py.
5. Add a clean_text(text) function that:
   - Removes repeated whitespace.
   - Removes excessive blank lines.
   - Keeps paragraph readability.
6. Update streamlit_app.py so that when a user uploads a file:
   - The file is parsed.
   - The text is cleaned.
   - The document text is stored in st.session_state.
7. Add an expander called "Preview Extracted Document Text".
8. Show character count and approximate word count.
9. Add clear error messages if extraction fails.
```

### Test

Upload:

```text
A PDF
A TXT file
A broken/empty file
```

### Commit

```bash
git add .
git commit -m "Add document upload and text extraction"
```

---

## Phase 3: Naive Chunking Baseline

### Goal

Build the baseline naive RAG pipeline for comparison.

### What you learn

```text
Fixed-size chunking
Chunk overlap
Baseline retrieval setup
Why naive RAG fails
```

### File to create

```text
src/naive_chunker.py
```

### Codex prompt

```text
Implement the naive chunking baseline.

Important:
- This phase should only implement fixed-size chunking.
- Do not implement semantic chunking yet.
- Add comments explaining why this is considered a naive baseline.

Requirements:
1. Create src/naive_chunker.py.
2. Add a function called naive_chunk_text(text, chunk_size=1000, overlap=150).
3. The function should split text by character count with overlap.
4. Each chunk should be returned as a dictionary with:
   - chunk_id
   - text
   - start_char
   - end_char
   - chunk_type set to "naive"
5. Update streamlit_app.py to:
   - Generate naive chunks after document upload.
   - Store them in st.session_state.
   - Display naive chunk count.
6. In Tab 3, add an expander showing naive chunks.
7. Add comments explaining the weakness of fixed-size chunking.
```

### Test

Check:

```text
Does the chunk count change when chunk size changes?
Are overlaps working?
Can you visually see sentence cuts?
```

### Commit

```bash
git add .
git commit -m "Add naive chunking baseline"
```

---

## Phase 4: Sentence Splitting for Semantic Chunking

### Goal

Prepare the document for semantic chunking.

### What you learn

```text
Sentence segmentation
Preprocessing for embeddings
Document structure preservation
```

### File to create

```text
src/semantic_chunker.py
```

### Codex prompt

```text
Start the semantic chunking pipeline by implementing sentence splitting.

Important:
- Do not implement embeddings yet.
- Focus only on reliable sentence extraction.
- Add comments explaining why sentence-level processing is needed.

Requirements:
1. Create src/semantic_chunker.py.
2. Add a function split_into_sentences(text).
3. The function should return a list of sentence dictionaries:
   - sentence_id
   - text
   - char_start
   - char_end
4. Avoid very tiny broken sentences where possible.
5. Update streamlit_app.py so that:
   - Sentences are generated after document upload.
   - Total sentence count is shown in Tab 1.
   - A preview table of first 20 sentences is shown in Tab 1.
6. Keep everything readable and suitable for learning.
```

### Commit

```bash
git add .
git commit -m "Add sentence splitting for semantic chunking"
```

---

## Phase 5: Sentence Embeddings and Semantic Distance

### Goal

Calculate distance between consecutive sentences.

### What you learn

```text
Embedding generation
Cosine similarity
Cosine distance
Topic shift detection
```

### Files to create

```text
src/embeddings.py
src/semantic_chunker.py update
```

### Codex prompt

```text
Add sentence embedding and semantic distance calculation.

Important:
- Use sentence-transformers for local embeddings.
- Use a small model suitable for Streamlit Community Cloud.
- Cache the embedding model using st.cache_resource in streamlit_app.py or a helper function.
- Add comments explaining what embeddings and cosine distance mean.

Requirements:
1. Create src/embeddings.py.
2. Add a function load_embedding_model(model_name).
3. Add a function embed_texts(texts, model).
4. Update src/semantic_chunker.py with:
   - calculate_sentence_distances(sentences, embeddings)
5. For each sentence pair, calculate:
   - sentence_id_current
   - sentence_id_next
   - cosine_similarity
   - cosine_distance
6. Update Tab 1 to show:
   - A dataframe of sentence distance values.
   - A line chart of cosine distance across the document.
7. Store sentence embeddings and distances in st.session_state.
```

### Test

Check:

```text
Do topic changes create visible spikes?
Does the chart appear in Tab 1?
Does caching stop the model from loading repeatedly?
```

Streamlit recommends `st.cache_resource` for global resources such as ML models and database connections, while `st.cache_data` is recommended for serializable data outputs. ([Streamlit Docs][3])

### Commit

```bash
git add .
git commit -m "Add sentence embeddings and semantic distance chart"
```

---

## Phase 6: Statistical Semantic Chunking

### Goal

Create breakpoints where topic shifts occur.

### What you learn

```text
Percentile thresholding
Standard deviation thresholding
Semantic breakpoint detection
Chunk construction
```

### Codex prompt

```text
Complete the statistical semantic chunking logic.

Important:
- Build semantic chunks using sentence distance spikes.
- Do not use arbitrary fixed character splitting for semantic chunks.
- Add detailed comments explaining the statistical threshold logic.

Requirements:
1. Update src/semantic_chunker.py.
2. Add a function find_semantic_breakpoints(distances, mode="percentile").
3. If mode is "percentile":
   - Use a high percentile such as 85th or 90th percentile.
4. If mode is "standard_deviation":
   - Use mean distance + one standard deviation.
5. Add a function build_semantic_chunks(sentences, breakpoints).
6. Each semantic chunk should include:
   - chunk_id
   - text
   - sentence_start
   - sentence_end
   - char_start
   - char_end
   - chunk_type set to "semantic"
7. Update Tab 1 to show:
   - Number of semantic chunks.
   - Breakpoint sentence IDs.
   - Semantic chunk table.
   - Semantic chunk previews.
8. Compare naive chunk count vs semantic chunk count.
```

### Test

Use a document with multiple topics.

Check:

```text
Are breakpoints meaningful?
Are semantic chunks readable?
Are complete ideas preserved better than naive chunks?
```

### Commit

```bash
git add .
git commit -m "Implement statistical semantic chunking"
```

---

## Phase 7: Metadata Anchoring

### Goal

Attach document-level context to each chunk.

### What you learn

```text
Chunk metadata
Context grounding
Source traceability
Document scope preservation
```

### File to create

```text
src/metadata_anchor.py
```

### Codex prompt

```text
Add metadata anchoring to both naive and semantic chunks.

Important:
- Metadata should make each chunk understandable outside the full document.
- Do not overcomplicate the system.
- Add comments explaining why metadata anchoring improves retrieval quality.

Requirements:
1. Create src/metadata_anchor.py.
2. Add a function create_document_metadata(file_name, full_text).
3. Metadata should include:
   - file_name
   - approximate_document_summary generated using the first few meaningful lines or a simple extractive summary
   - total_characters
   - total_words
4. Add a function attach_metadata_to_chunks(chunks, metadata).
5. Each chunk should receive:
   - file_name
   - document_summary
   - source_type
6. Add a function build_embedding_text(chunk) that combines:
   - document title or file name
   - document summary
   - chunk text
7. Update the app so chunks are embedded with metadata-anchored text, but displayed to the user with clean chunk text.
8. Show metadata examples in Tab 1.
```

### Interview point

```text
I did not embed isolated chunks only. I embedded chunks with document-level metadata, so each vector carries local meaning plus global document scope.
```

### Commit

```bash
git add .
git commit -m "Add metadata anchoring for chunks"
```

---

## Phase 8: Parent-Child Retrieval Structure

### Goal

Search smaller child chunks but pass larger parent chunks to the LLM.

### What you learn

```text
Parent-child retrieval
Chunk hierarchy
Precision vs context trade-off
```

### File to create

```text
src/parent_child.py
```

### Codex prompt

```text
Implement parent-child retrieval data structures.

Important:
- Child chunks should be used for precise search.
- Parent chunks should be used for final answer context.
- Add comments explaining the difference between parent and child chunks.

Requirements:
1. Create src/parent_child.py.
2. Add a function create_parent_child_chunks(semantic_chunks).
3. Treat each semantic chunk as a parent chunk.
4. Split each parent chunk into smaller child chunks around 150-250 tokens.
5. Each child chunk should include:
   - child_id
   - parent_id
   - child_text
   - parent_text
   - metadata
6. Update the app so the vector index is built from child chunks.
7. Store a mapping from child_id to parent chunk.
8. In Tab 1, display:
   - Parent chunk count
   - Child chunk count
   - Sample parent-child mapping
```

### Interview point

```text
The retriever searches small child chunks for precision, but the generator receives the larger parent section for context completeness.
```

### Commit

```bash
git add .
git commit -m "Add parent-child retrieval structure"
```

---

## Phase 9: Vector Store Retrieval

### Goal

Retrieve top 10 raw vector hits.

### What you learn

```text
Embedding index
FAISS retrieval
Similarity search
Raw retrieval ranking
```

### File to create

```text
src/vector_store.py
```

### Codex prompt

```text
Implement vector search over child chunks.

Important:
- Use FAISS or a simple numpy similarity search if FAISS causes setup issues.
- Keep the retrieval logic easy to understand.
- Add comments explaining the vector search flow.

Requirements:
1. Create src/vector_store.py.
2. Add a function build_vector_index(child_chunks, embedding_model).
3. Add a function retrieve_top_k(query, child_chunks, index, embedding_model, top_k=10).
4. Each retrieval result should include:
   - raw_rank
   - child_id
   - parent_id
   - child_text
   - parent_text
   - vector_similarity
   - metadata
5. Update streamlit_app.py so that when the user clicks Run Analysis:
   - The query is embedded.
   - Top 10 raw vector hits are retrieved.
   - Results are stored in session state.
6. In Tab 2, show the raw vector hits table.
```

### Test

Ask a question that should clearly match one section.

Check:

```text
Does the correct section appear in the top 10?
Are irrelevant chunks also appearing?
Can you explain why re-ranking is needed?
```

### Commit

```bash
git add .
git commit -m "Add vector retrieval over child chunks"
```

---

## Phase 10: Cross-Encoder Re-ranking

### Goal

Re-rank the top 10 vector hits and keep only the best 3.

### What you learn

```text
Candidate generation
Cross-encoder relevance scoring
Ranking comparison
Precision filtering
```

### File to create

```text
src/reranker.py
```

### Codex prompt

```text
Add cross-encoder re-ranking.

Important:
- Use a small cross-encoder model suitable for local development and Streamlit deployment.
- Cache the re-ranker model using st.cache_resource.
- Add comments explaining why cross-encoders are more precise than bi-encoder vector search.

Requirements:
1. Create src/reranker.py.
2. Add a function load_reranker_model(model_name).
3. Add a function rerank_results(query, raw_results, reranker_model, top_n=3).
4. The function should:
   - Take the top 10 raw vector results.
   - Score each query + chunk pair.
   - Sort by re-ranker score.
   - Return the top 3.
5. Each result should include:
   - raw_rank
   - rerank_position
   - child_id
   - parent_id
   - vector_similarity
   - reranker_score
   - selected
   - text preview
6. Update Tab 2 to show:
   - Raw Top 10 table
   - Final Top 3 table
   - Raw rank vs re-ranked position
7. Add a short explanation box in Tab 2 explaining the latency vs precision trade-off.
```

### Interview point

```text
Vector search narrows the candidate pool quickly, then the cross-encoder performs deeper query-document relevance scoring before anything reaches the LLM.
```

### Commit

```bash
git add .
git commit -m "Add cross-encoder reranking analytics"
```

---

## Phase 11: Context Builder and Token Metrics

### Goal

Build final LLM context and calculate token savings.

### What you learn

```text
Prompt construction
Token counting
Context compression
Retrieval efficiency metrics
```

### Files to create

```text
src/context_builder.py
src/metrics.py
```

### Codex prompt

```text
Implement engineered context building and token-efficiency metrics.

Important:
- The goal is to show how much context is saved compared to naive RAG and the full document.
- Add comments explaining each metric.

Requirements:
1. Create src/context_builder.py.
2. Add a function build_naive_context(raw_vector_results, max_chunks=5).
3. Add a function build_engineered_context(reranked_results).
4. Engineered context should use parent_text from the selected parent chunks.
5. Avoid duplicate parent chunks.
6. Create src/metrics.py.
7. Add approximate token counting.
8. Add a function calculate_context_metrics(full_text, naive_context, engineered_context).
9. Metrics should include:
   - full_document_tokens
   - naive_context_tokens
   - engineered_context_tokens
   - tokens_saved_percent
   - context_reduction_percent
10. Update the metric cards at the top of the app.
11. In Tab 3, show the exact context used by naive RAG and engineered context inside expanders.
```

### Test

Check:

```text
Do token numbers make sense?
Does engineered context use fewer tokens?
Are duplicate parent chunks removed?
```

### Commit

```bash
git add .
git commit -m "Add context builder and token metrics"
```

---

## Phase 12: LLM Answer Generation

### Goal

Generate side-by-side answers.

### What you learn

```text
LLM prompting
Grounded generation
Prompt comparison
Naive vs engineered RAG
```

### File to create

```text
src/llm_client.py
```

### Codex prompt

```text
Add LLM answer generation for naive RAG and engineered context.

Important:
- Use an API key from Streamlit secrets.
- Do not hardcode API keys.
- Keep prompts clear and grounded.
- Add comments explaining the prompt structure.

Requirements:
1. Create src/llm_client.py.
2. Add a function generate_answer(question, context, mode).
3. The function should:
   - Use the selected LLM provider.
   - Answer only using the given context.
   - Say when the context is insufficient.
   - Return a structured answer.
4. Update streamlit_app.py so that:
   - Naive context is sent to generate the naive answer.
   - Engineered context is sent to generate the engineered answer.
5. In Tab 3, display answers side by side.
6. Add labels:
   - "Naive RAG Answer"
   - "Engineered Context Answer"
7. Add an expander under each answer showing the context used.
8. Use st.secrets for API keys.
```

### Secret example

For local development:

```toml
OPENAI_API_KEY = "your_key_here"
```

For deployment, Streamlit’s Community Cloud docs recommend storing credentials through the app’s secrets management rather than committing them into the repository. ([Streamlit Docs][4])

### Commit

```bash
git add .
git commit -m "Add side-by-side LLM answer generation"
```

---

## Phase 13: Final UI Polish

### Goal

Make it look like a professional portfolio app.

### What you learn

```text
UX polish
Streamlit layout refinement
Readable analytics
Portfolio presentation
```

### File to create

```text
src/ui_components.py
```

### Codex prompt

```text
Polish the Streamlit UI for portfolio presentation.

Important:
- Do not change the core backend logic.
- Improve readability and presentation.
- Add comments where useful.

Requirements:
1. Create src/ui_components.py.
2. Add reusable UI functions for:
   - metric card display
   - explanation boxes
   - chunk preview cards
   - warning/info messages
3. Improve the sidebar with a "How this works" section.
4. Add a pipeline summary:
   Naive RAG:
   Fixed chunks → vector search → LLM

   Engineered Context:
   Semantic chunks → metadata anchoring → parent-child retrieval → vector search → re-ranking → compact context → LLM
5. In Tab 1, make the chunk map easier to read.
6. In Tab 2, highlight selected top 3 re-ranked rows.
7. In Tab 3, make the answer comparison clean and readable.
8. Add a final "Interview Talking Points" expander.
```

### Commit

```bash
git add .
git commit -m "Polish UI for portfolio presentation"
```

---

## Phase 14: README Documentation

### Goal

Create a strong GitHub README.

### What you learn

```text
Technical writing
Portfolio documentation
Architecture explanation
Interview positioning
```

### Codex prompt

```text
Write a professional README.md for this project.

Important:
- This README is for recruiters, interviewers, and technical reviewers.
- Do not exaggerate results.
- Explain the project clearly.

README sections required:
1. Project title
2. Short summary
3. Problem with naive RAG
4. What this app demonstrates
5. Features
6. Architecture diagram in text form
7. Technical pipeline
8. Streamlit tabs explained
9. Metrics explained
10. Tech stack
11. Local setup instructions
12. Environment variables / secrets
13. Deployment instructions
14. Interview talking points
15. Future improvements
```

### Commit

```bash
git add .
git commit -m "Add professional README"
```

---

## Phase 15: requirements.txt and Deployment Preparation

### Goal

Prepare the app for Streamlit Community Cloud.

### What you learn

```text
Dependency management
Cloud deployment constraints
Secrets management
Resource optimization
```

### Suggested requirements

```text
streamlit
pandas
numpy
scikit-learn
sentence-transformers
pymupdf
faiss-cpu
openai
tiktoken
```

If FAISS causes deployment issues, use numpy similarity search first, then add FAISS later.

Streamlit Community Cloud supports dependency installation from files like `requirements.txt`; the docs explain that dependency files are searched from the app entrypoint directory and repository root. ([Streamlit Docs][1])

### Codex prompt

```text
Prepare the project for Streamlit Community Cloud deployment.

Important:
- Do not add unnecessary heavy dependencies.
- Keep the app deployment-friendly.
- Add comments where useful.

Requirements:
1. Update requirements.txt with only the required packages.
2. Ensure all imports in the project are included in requirements.txt.
3. Add a .gitignore that excludes:
   - venv
   - __pycache__
   - .streamlit/secrets.toml
   - local data files if needed
4. Add .streamlit/config.toml with a clean theme.
5. Ensure the app can run from streamlit_app.py.
6. Add graceful fallback messages if a model fails to load.
7. Use caching for embedding and re-ranker models.
8. Add a warning in the UI for very large uploaded documents.
```

### Commit

```bash
git add .
git commit -m "Prepare app for Streamlit deployment"
```

---

# 10. Deployment on Streamlit Community Cloud

## Step 1: Push final code

```bash
git add .
git commit -m "Final deployment-ready version"
git push
```

## Step 2: Open Streamlit Community Cloud

Go to:

```text
share.streamlit.io
```

Streamlit’s official getting-started docs describe deployment as three broad steps: put the app in a public GitHub repo with requirements, sign into Streamlit Community Cloud, and deploy the app from the GitHub URL. ([Streamlit Docs][5])

## Step 3: Create app

Select:

```text
Repository: advanced-context-engineering-harness
Branch: main
Main file path: streamlit_app.py
```

## Step 4: Add secrets

In Streamlit app settings, add:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

## Step 5: Deploy

After deployment, test:

```text
1. Upload TXT file
2. Upload PDF file
3. Ask a complex question
4. Check all three tabs
5. Check token metrics
6. Check answer comparison
7. Check app logs if anything fails
```

Streamlit also provides app management options such as rebooting, viewing analytics, settings, and cloud logs from the Community Cloud workspace. ([Streamlit Docs][6])

---

# 11. Important Deployment Warnings

## Avoid very heavy models at first

For learning and deployment, start with small models.

Use:

```text
sentence-transformers/all-MiniLM-L6-v2
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Avoid starting with very large re-rankers because they may load slowly or consume too much memory.

Streamlit’s resource-limit guidance recommends using caching, limiting cache size, moving big datasets outside the app, and profiling memory usage when apps become resource-heavy. ([Streamlit Docs][7])

---

# 12. What to Show in Your Portfolio Demo

Use a demo script like this:

```text
First, I upload a long technical document.

Then I ask a complex question.

The app first shows naive chunking, where the document is split mechanically.

Next, the Semantic Chunk Map shows where the document actually changes meaning.

Then, the Re-ranker Analytics tab shows that vector search retrieved 10 candidates, but the cross-encoder selected only the best 3.

Finally, the comparison tab shows that the engineered context produces a more focused answer using fewer tokens.
```

---

# 13. Interview Deep-Dive Explanation

## When asked: “Why not just use normal RAG?”

Say:

```text
Normal RAG often assumes that fixed-size chunks are good enough. But fixed chunking can split ideas across boundaries, which weakens retrieval quality. My app demonstrates a more engineered approach where chunks are created based on semantic shifts, not arbitrary token counts.
```

## When asked: “Why do you need re-ranking?”

Say:

```text
Vector search is fast but approximate. It retrieves candidates based on embedding similarity, but it may rank shallow matches too highly. The cross-encoder re-ranker evaluates the query and document together, giving a deeper relevance score before the final context is sent to the LLM.
```

## When asked: “What is the business value?”

Say:

```text
The business value is lower cost, lower latency at generation time, and better answer quality. By sending fewer but more relevant tokens to the LLM, the system reduces prompt size while preserving the evidence needed for accurate answers.
```

## When asked: “What trade-off did you make?”

Say:

```text
The re-ranker adds extra retrieval latency, but it reduces the final LLM prompt size. In many workflows, that trade-off is worth it because generation becomes cheaper and the final answer is better grounded.
```

---

# 14. Final Learning Path

Build it in this order:

```text
1. UI skeleton
2. Document extraction
3. Naive chunking
4. Sentence splitting
5. Sentence embeddings
6. Semantic distance chart
7. Semantic chunking
8. Metadata anchoring
9. Parent-child retrieval
10. Vector search
11. Cross-encoder re-ranking
12. Token metrics
13. Naive vs engineered answer comparison
14. UI polish
15. README
16. Deployment
```

---

# 15. Final Git Commit Sequence

Use this sequence to show professional development history:

```bash
git add .
git commit -m "Create Streamlit UI skeleton"

git add .
git commit -m "Add document upload and text extraction"

git add .
git commit -m "Add naive chunking baseline"

git add .
git commit -m "Add sentence splitting for semantic chunking"

git add .
git commit -m "Add sentence embeddings and semantic distance chart"

git add .
git commit -m "Implement statistical semantic chunking"

git add .
git commit -m "Add metadata anchoring for chunks"

git add .
git commit -m "Add parent-child retrieval structure"

git add .
git commit -m "Add vector retrieval over child chunks"

git add .
git commit -m "Add cross-encoder reranking analytics"

git add .
git commit -m "Add context builder and token metrics"

git add .
git commit -m "Add side-by-side LLM answer generation"

git add .
git commit -m "Polish UI for portfolio presentation"

git add .
git commit -m "Add professional README"

git add .
git commit -m "Prepare app for Streamlit deployment"
```

---

# 16. Final Project Outcome

By the end, you will have a portfolio project that demonstrates:

```text
Semantic chunking
Naive RAG comparison
Embedding-based retrieval
Metadata anchoring
Parent-child retrieval
Cross-encoder re-ranking
Token-efficiency metrics
Context reduction analysis
Streamlit deployment
Clean GitHub documentation
Interview-ready technical storytelling
```

This is much stronger than saying:

```text
I built a RAG chatbot.
```

Instead, you can say:

```text
I built a context engineering evaluation harness that visualizes and measures how engineered retrieval reduces token bloat and improves context precision compared with naive RAG.
```

[1]: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies?utm_source=chatgpt.com "App dependencies for your Community Cloud app"
[2]: https://docs.streamlit.io/develop/api-reference/connections/st.secrets?utm_source=chatgpt.com "st.secrets - Streamlit Docs"
[3]: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts?utm_source=chatgpt.com "Advanced concepts of Streamlit"
[4]: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management?utm_source=chatgpt.com "Secrets management for your Community Cloud app"
[5]: https://docs.streamlit.io/get-started/tutorials/create-an-app?utm_source=chatgpt.com "Create an app - Streamlit Docs"
[6]: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app?utm_source=chatgpt.com "Manage your app - Streamlit Docs"
[7]: https://docs.streamlit.io/knowledge-base/deploy/resource-limits?utm_source=chatgpt.com "Argh. This app has gone over its resource limits"
