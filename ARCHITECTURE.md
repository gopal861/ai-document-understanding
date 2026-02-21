# AI Document Understanding System — Architecture

**Author:** Gopal Khandare
**Date:** February 2026
**Status:** Production
**Version:** 1.0
**Deployment:** https://ai-document-understanding.onrender.com
**Deployment:** https://gopal861-ai-document-understanding.hf.space
**Runtime:** Python 3.11.9 · FastAPI · Uvicorn · Render ·Huggingface 

---

## 1. System Overview

The AI Document Understanding System is a production-grade Retrieval-Augmented Generation (RAG) API. It enables users to upload documents from multiple sources — PDFs, websites, GitHub repositories, and markdown files — and ask natural language questions against them. The system retrieves the most relevant document sections, constructs a grounded prompt, and generates an accurate answer using a multi-provider LLM backend.

The system is designed around three core guarantees:

- **Grounded answers only** — every answer is backed by retrieved document content. If context is insufficient, the system refuses rather than guesses.
- **Persistence across restarts** — document vectors survive instance restarts via cloud vector storage with automatic local rebuild on startup.
- **Provider resilience** — automatic LLM fallback from OpenAI to Gemini ensures continuity through provider outages.

---

## 2. Core Components

### 2.1 HTTP Layer — `main.py`

**Responsibility:** Application entry point. Configures the FastAPI application, registers middleware, and handles cross-cutting concerns.

**Inputs:** Raw HTTP requests from any client.

**Outputs:** Structured HTTP responses. Unhandled exceptions converted to `500` JSON responses with `request_id`.

**Key behaviors:**
- CORS middleware configured with open origins (`allow_origins=["*"]`) for API accessibility.
- Per-request UUID injected into `request.state.request_id` at middleware level. This ID is carried through all downstream logging and event tracking calls.
- Global exception handler catches any unhandled exception, logs it with `request_id`, tracks it in PostHog, and returns a structured error response.
- Startup event triggers `load_document_registry()` to restore document metadata from disk.

**Dependencies:** `routes.py`, `logger.py`, `metrics.py`, `posthog_client.py`

---

### 2.2 API Layer — `routes.py`

**Responsibility:** Route definitions, request validation, response serialization, and ingestion orchestration.

**Inputs:** Validated HTTP requests (Pydantic models).

**Outputs:** Typed response models (`UploadResponse`, `AskResponse`, `ListDocumentsResponse`, etc.).

**Endpoints:**

| Method | Path | Responsibility |
|---|---|---|
| `POST` | `/upload` | Accept file or URL, run ingestion pipeline, register document |
| `POST` | `/ask` | Accept question + doc_id, run query pipeline, return answer |
| `GET` | `/documents` | Return all registered documents with metadata |
| `DELETE` | `/documents/{id}` | Remove document from registry |
| `GET` | `/health` | Return system health and vector store statistics |
| `GET` | `/metrics` | Return request metrics and latency percentiles |
| `GET` | `/` | API information |

**Singletons initialized at module load:**
- `Embedder()` — one instance shared across all requests
- `VectorStore(dim=1536)` — one instance, triggers Qdrant rebuild if local state is empty
- `MultiModelLLMClient()` — one instance, initializes both OpenAI and Gemini at startup

**Document registry:** In-memory `Dict[str, dict]` backed by `storage/document_registry.json`. Loaded from disk at startup. Rebuilt from Qdrant stats if disk file is missing.

**Ingestion concurrency:** `threading.Lock()` wraps all `vector_store.add()` calls to prevent concurrent writes to the shared FAISS index.

**Dependencies:** All memory, LLM, workflow, and observability modules.

---

### 2.3 Ingestion Pipeline

The ingestion pipeline transforms a raw document source into indexed, searchable vector embeddings. It is composed of four sequential components.

#### 2.3.1 Loader — `loader.py`

**Responsibility:** Extract clean text from any supported document source.

**Inputs:** File path (PDF, Markdown) or URL (HTML, GitHub, raw Markdown).

**Outputs:** Plain text string, enforced under `MAX_DOCUMENT_CHARACTERS = 2,000,000`.

**Design contract:** Fully synchronous. No `asyncio` inside this module. Async execution is handled at the routes layer via `asyncio.run_in_executor(None, load_text, source)`.

**Source routing:**

```
source is URL:
    domain == raw.githubusercontent.com → HTTP GET, return text directly
    domain == github.com                → GitHub API tree traversal, collect .md/.py/.ipynb files
    else:
        run static HTML crawler (BeautifulSoup, recursive, max 20 pages)
        if result < 1500 chars:
            run Playwright chromium headless fallback
            return whichever result is longer

source ends with .pdf  → pypdf PdfReader, extract all pages
source ends with .md   → read file directly
```

**URL security:** All URLs validated against `ALLOWED_DOMAINS` allowlist before any HTTP request is made. Requests to unlisted domains raise `ValueError`.

**Playwright configuration:** Chromium launched with `--no-sandbox`, `--disable-dev-shm-usage`, `--disable-gpu`. Browser path set to `/opt/render/project/.playwright` for Render compatibility.

**Retry:** Loader is called through `load_text_with_retry()` in routes — 3 attempts with 2s delay between attempts.

---

#### 2.3.2 Chunker — `chunker.py`

**Responsibility:** Split extracted text into overlapping fixed-size chunks suitable for embedding.

**Inputs:** Plain text string.

**Outputs:** `List[str]` of text chunks.

**Algorithm:** Sliding window over word tokens.

```
words = text.split()
step = CHUNK_SIZE - CHUNK_OVERLAP  →  700 - 120 = 580 words
for each window [start : start + 700]:
    chunk = " ".join(words[start:end])
    start += step
```

**Parameters:**
- `CHUNK_SIZE = 700` words per chunk
- `CHUNK_OVERLAP = 120` words of overlap between consecutive chunks
- Step = 580 words (non-overlapping stride)

**Safety guarantees:** Empty text rejected early. Text truncated at `MAX_DOCUMENT_CHARACTERS` before chunking. Overlap validated to be less than chunk size. Infinite loop prevention via `step <= 0` guard.

---

#### 2.3.3 Embedder — `embedder.py`

**Responsibility:** Convert text chunks into L2-normalized vector embeddings.

**Inputs:** `List[str]` of text chunks.

**Outputs:** `numpy.ndarray` of shape `(n_chunks, 1536)`, dtype `float32`, L2-normalized.

**Model:** OpenAI `text-embedding-3-small` → 1536 dimensions.

**Batching:** Chunks sent to OpenAI API in batches of 32. Each batch normalized immediately after receipt. All batches concatenated via `np.vstack()`.

**Normalization:** L2 normalization applied per batch:
```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms
```
Normalization enables cosine similarity to be computed via dot product in FAISS `IndexFlatIP`.

**Limit enforcement:** `MAX_CHUNKS_PER_DOCUMENT = 500` checked before any API call. Raises `ValueError` if exceeded.

---

#### 2.3.4 Vector Store — `store.py`

**Responsibility:** Persist embeddings and chunk metadata. Serve vector similarity queries filtered by document.

**Inputs (write):** Embeddings array, chunk text list, doc_id string.

**Outputs (read):** `List[Dict]` with `text`, `doc_id`, `chunk_idx`, `similarity_score`.

**Dual storage architecture:**

| Layer | Technology | Role |
|---|---|---|
| Cloud persistence | Qdrant | Primary durable store. All writes go here first. |
| Local cache | FAISS `IndexFlatIP` | In-memory index. Rebuilt from Qdrant on startup. |
| Chunk metadata | `storage/metadata.json` | Maps global index to chunk text and doc_id. |

**Write path:**
```
vector_store.add(embeddings, chunks, doc_id):
    1. Normalize embeddings
    2. Build PointStruct list (uuid4 id, vector, payload: {text, doc_id, chunk_idx})
    3. Qdrant upsert → qdrant_client.upsert(collection, points)
    4. FAISS index.add(embeddings)
    5. Append to _chunks list
    6. Update _doc_chunk_count
    7. _save_to_disk() → write faiss.index + metadata.json
```

**Query path:**
```
vector_store.query(embedding, top_k=8, doc_id=filter):
    1. Normalize query embedding
    2. Qdrant search with doc_id payload filter
    3. Return [{text, doc_id, chunk_idx, similarity_score}]
```

**Startup rebuild sequence:**
```
VectorStore.__init__():
    _load_from_disk()           # attempt local FAISS + metadata
    if len(_chunks) == 0:
        _rebuild_from_qdrant()  # paginated scroll, 100 points/page
                                # rebuilds _chunks, _doc_chunk_count, FAISS
        _save_to_disk()         # restore local cache
```

**Qdrant collection:** Cosine distance, 1536 dimensions. Created automatically if not present on first startup.

---

### 2.4 Query Pipeline

#### 2.4.1 Retriever — `retriever.py`

**Responsibility:** Generate query embedding and execute vector similarity search.

**Inputs:** Question string, embedder instance, store instance, `top_k`, optional `doc_id` filter.

**Outputs:** `List[Dict]` from `store.query()` — ranked by similarity score descending.

**Execution:**
```python
query_embedding = embedder.embed([question])   # 1×1536 normalized vector
results = store.query(query_embedding, top_k, doc_id)
return results
```

---

#### 2.4.2 Document QA Workflow — `document_qa.py`

**Responsibility:** Orchestrate the complete question-answering pipeline from retrieval to response.

**Inputs:** `question`, `session_id` (doc_id), `retrieve_fn`, `llm_client`, `top_k`.

**Outputs:** Structured dict with `answer`, `document_id`, `confidence_score`, `refused`, `sources_used`, `reasoning`.

**Pipeline:**

```
Step 1 — Retrieve
    context_chunks = retrieve_fn(question, top_k=8)
    top_score = context_chunks[0]["similarity_score"]
    PostHog: track_retrieval(chunks_retrieved, top_score)

Step 2 — Empty context guard
    if not context_chunks:
        return refused=True, confidence=0.0

Step 3 — Similarity threshold gate
    if top_score < SIMILARITY_THRESHOLD (0.45):
        return refused=True, confidence=top_score

Step 4 — Prompt construction
    prompt = build_document_prompt(question, context_chunks, model_type="cloud")

Step 5 — LLM generation
    answer = llm_client.generate(prompt)
    PostHog: track llm_completed (latency, chunks_used, top_score)

Step 6 — Return response
    return answer, confidence_score, refused=False, sources_used
```

---

### 2.5 LLM Client — `multi_model_client.py`

**Responsibility:** Multi-provider LLM generation with automatic fallback.

**Inputs:** Formatted prompt string.

**Outputs:** Generated answer string.

**Provider initialization (at startup):**
- OpenAI: key validated with `sk-` prefix check
- Gemini: initialized via `genai.Client(api_key=key)`, validated with live `"ping"` test call
- Availability flags set: `openai_available`, `gemini_available`

**Fallback logic:**
```
generate(prompt):
    if openai_available → _generate_openai(prompt)
        success → return result
        failure → log warning, fall through

    if gemini_available → _generate_gemini(prompt)
        success → return result
        failure → log warning

    raise RuntimeError("No LLM backend available")
```

**OpenAI call:** `gpt-4o-mini`, `temperature=0.1`, `max_tokens=500`. System prompt injected as `role: system` message.

**Gemini call:** `gemini-2.0-flash`. System prompt prepended to user content as single string.

**Latency tracking:** Every call wrapped in `_timed_call()` — logs `provider` and `latency_seconds` on success.

---

### 2.6 Prompt Layer — `prompt_builder.py`, `system_prompts.py`

**Responsibility:** Construct grounded prompts that prevent hallucination and enforce document-only answers.

**Design rule:** No prompts hardcoded in workflow or LLM client files. All prompts imported from `system_prompts.py`.

**Context block construction:**
```
[Context 1 | Confidence: 0.823]
<chunk text from retrieval>

[Context 2 | Confidence: 0.791]
<chunk text from retrieval>
...
```

**System prompt behavior (enforced in `DOCUMENT_QA_SYSTEM_PROMPT`):**
- Answer using ONLY the provided document context
- Combine information across multiple chunks to form complete answers
- Exact refusal phrase defined: `"I don't have enough information in the document to answer this."`
- Do not use outside knowledge
- Do not reference the context or chunks in the answer
- Priority order: Accuracy → Grounded reasoning → Completeness → Conciseness

**Model type routing:** `model_type="cloud"` uses `DOCUMENT_QA_SYSTEM_PROMPT`. `model_type="local"` uses `LOCAL_MODEL_SYSTEM_PROMPT` (shorter, deterministic).

---

### 2.7 Observability Stack

Three independent observability layers operate in parallel. None blocks request execution.

#### 2.7.1 JSON Logger — `logger.py`

**Responsibility:** Structured logging with safe extra field injection.

**Output format:** JSON per log line with: `timestamp`, `level`, `logger`, `message`, `module`, `function`, `line`, plus any custom `extra` fields passed by the caller.

**Handlers:** Console (stdout) + file (`logs/app.log`). Both use `JSONFormatter`.

**Reserved field protection:** Custom extra fields that collide with standard `LogRecord` attributes are prefixed with `extra_` to prevent overwriting.

**Noise suppression:** `urllib3`, `httpx`, `openai` loggers set to `WARNING` to suppress client library verbosity.

---

#### 2.7.2 Metrics Tracker — `metrics.py`

**Responsibility:** In-process request metrics with percentile computation.

**Tracked fields:**
- `total_requests`, `successful_requests`, `failed_requests`
- `total_latency`, `avg_latency`
- `latencies` — full list of individual request latencies for percentile calculation

**Thread safety:** All writes protected by `threading.Lock()`.

**Persistence:** Written to `storage/metrics.json` on every update. Survives process restart via disk load at init.

**Percentile API:** `get_latency_percentile(p)` — sorts latency list, returns value at index `int(n * p / 100)`.

**Exposure:** `GET /metrics` returns full metrics dict.

---

#### 2.7.3 PostHog Client — `posthog_client.py`

**Responsibility:** Production event tracking for usage analytics and error monitoring.

**Events tracked:**

| Event | Trigger | Properties |
|---|---|---|
| `document_uploaded` | Successful upload | doc_id, filename, chunks, latency |
| `question_asked` | Ask endpoint | doc_id, question_length, latency, success |
| `retrieval_completed` | After vector search | doc_id, chunks_retrieved, top_score |
| `llm_completed` | After generation | latency, workflow_latency, chunks_used, top_score |
| `system_error` | Any exception path | error_type, error_message, endpoint |

**Identity model:** `request_id` (UUID) used as `distinct_id`. Maps naturally to future user identity when authentication is added.

**Reliability:** All `_track()` calls wrapped in try/except. PostHog failure never propagates to request handler.

**Configuration:** `flush_interval=1` for near-real-time dashboard updates. `timeout=5s` on PostHog client.

---

## 3. Data Flow

### 3.1 Document Ingestion Flow

```
User → POST /upload (file or URL)
  │
  ├─ Generate doc_id: "doc_" + uuid4().hex[:12]
  │
  ├─ load_text_with_retry(source)          [async executor, 3 retries]
  │     └─ loader.load_text(source)        [sync, PDF/HTML/GitHub/MD]
  │
  ├─ chunker.chunk_text(text)              [700 words, 120 overlap]
  │
  ├─ embedder.embed(chunks)               [OpenAI API, batched 32, normalized]
  │
  ├─ with ingestion_lock:
  │     vector_store.add(embeddings, chunks, doc_id)
  │           ├─ Qdrant upsert            [cloud, persistent]
  │           ├─ FAISS index.add          [local, in-memory]
  │           └─ metadata.json write      [local disk]
  │
  ├─ document_registry[doc_id] = {filename, chunks_count, timestamp}
  ├─ save_document_registry()             [storage/document_registry.json]
  │
  ├─ posthog_client.track_document_upload(...)
  │
  └─ UploadResponse(document_id, filename, chunks_created)
```

---

### 3.2 Question Answering Flow

```
User → POST /ask (document_id, question)
  │
  ├─ Validate document_id in document_registry
  │
  ├─ document_qa.answer_question(question, doc_id, retrieve_fn, llm_client)
  │
  │   ├─ retriever.retrieve(question, embedder, store, top_k=8, doc_id)
  │   │     ├─ embedder.embed([question])     [1×1536 normalized vector]
  │   │     └─ store.query(embedding, 8, doc_id)
  │   │           └─ Qdrant search with doc_id payload filter
  │   │
  │   ├─ posthog_client.track_retrieval(...)
  │   │
  │   ├─ [Gate 1] if no chunks → return refused=True
  │   ├─ [Gate 2] if top_score < 0.45 → return refused=True
  │   │
  │   ├─ prompt_builder.build_document_prompt(question, chunks)
  │   │     └─ DOCUMENT_QA_SYSTEM_PROMPT + context block + question
  │   │
  │   ├─ llm_client.generate(prompt)
  │   │     ├─ OpenAI gpt-4o-mini (primary)
  │   │     └─ Gemini gemini-2.0-flash (fallback)
  │   │
  │   ├─ posthog_client.track llm_completed(...)
  │   │
  │   └─ return {answer, confidence_score, refused, sources_used}
  │
  └─ AskResponse(answer, document_id, confidence_score, refused, sources_used)
```

---

## 4. Storage Layer

### 4.1 Storage Map

| Data | Technology | Location | Durability |
|---|---|---|---|
| Vectors + payloads | Qdrant Cloud | Cloud | Permanent |
| FAISS index | File | `storage/faiss.index` | Instance lifetime |
| Chunk metadata | JSON file | `storage/metadata.json` | Instance lifetime |
| Document registry | JSON file | `storage/document_registry.json` | Instance lifetime |
| Request metrics | JSON file | `storage/metrics.json` | Instance lifetime |
| Application logs | JSON file | `logs/app.log` | Instance lifetime |

### 4.2 Qdrant Point Schema

```json
{
  "id": "<uuid4>",
  "vector": [1536 × float32],
  "payload": {
    "text": "<chunk text>",
    "doc_id": "doc_abc123def456",
    "chunk_idx": 0
  }
}
```

Collection: `documents`. Distance: Cosine. Dimension: 1536.
Auto-created on first startup if not present.

### 4.3 Startup State Recovery

On every startup, the system executes:

```
1. load_document_registry()
   → read storage/document_registry.json
   → restore in-memory document_registry dict

2. VectorStore.__init__()
   → _load_from_disk()
   → if chunks == 0: _rebuild_from_qdrant()
       → scroll Qdrant, 100 points per page
       → reconstruct _chunks, _doc_chunk_count, FAISS index
       → write back to disk

3. rebuild_registry_from_vector_store()
   → reconcile document_registry with vector store stats
   → add any doc_ids present in Qdrant but missing from registry
```

This sequence ensures the system is fully operational within seconds of startup, regardless of whether local disk state exists.

---

## 5. LLM Interaction Layer

### 5.1 Model Configuration

| Property | Value |
|---|---|
| Primary model | `gpt-4o-mini` (OpenAI) |
| Fallback model | `gemini-2.0-flash` (Google) |
| Temperature | 0.1 (low — factual answers) |
| Max tokens | 500 |
| Embedding model | `text-embedding-3-small` |
| Embedding dimensions | 1536 |

### 5.2 Prompt Flow

```
system_prompts.py
    DOCUMENT_QA_SYSTEM_PROMPT
            │
            ▼
prompt_builder.build_document_prompt(question, context_chunks)
    ├─ system prompt
    ├─ context block (N chunks with confidence scores)
    ├─ question
    └─ answer instruction
            │
            ▼
llm_client.generate(prompt)
    ├─ OpenAI: {role: system, content: SYSTEM_PROMPT}
    │          {role: user, content: full prompt}
    └─ Gemini: SYSTEM_PROMPT + "\n\n" + full prompt
```

### 5.3 Hallucination Prevention

Three independent gates prevent out-of-context answers:

**Gate 1 — Empty retrieval (pre-LLM)**
No chunks retrieved → immediate refusal. LLM never called. Zero token cost.

**Gate 2 — Similarity threshold (pre-LLM)**
`top_score < 0.45` → immediate refusal. LLM never called. Zero token cost.

**Gate 3 — Prompt-level grounding (in-LLM)**
System prompt explicitly instructs the model to answer only from provided context and return the defined refusal phrase if the answer is not present.

**Measured outcome:** 96% answer accuracy, 100% grounded answer rate across 25 production queries across 3 real documents.

---

## 6. Scalability Analysis

### 6.1 Current Architecture Limits

| Dimension | Current Limit | Constraint |
|---|---|---|
| Documents in memory | 100 | `MAX_DOCUMENTS_IN_MEMORY` config |
| Chunks per document | 500 | `MAX_CHUNKS_PER_DOCUMENT` config |
| Total vectors | Qdrant plan limit | Cloud tier dependent |
| Concurrent uploads | Serialized via lock | `ingestion_lock` |
| Concurrent queries | Unbounded | Qdrant handles concurrency |

### 6.2 Bottlenecks

**Ingestion bottleneck:** `ingestion_lock` serializes FAISS writes. Concurrent uploads queue behind the lock. Qdrant upserts happen before the lock and are concurrent-safe.

**Query bottleneck:** Two sequential external API calls per query — OpenAI embedding + OpenAI (or Gemini) generation. Each adds 1–6s of network latency. No caching in current design.

**Memory bottleneck:** FAISS index is fully in-memory. At 1536 dimensions × 4 bytes × 50,000 vectors = ~308MB RAM. Render free tier has 512MB RAM limit.

### 6.3 Horizontal Scale Path

The system is stateless at the request level. State is held in:
- Qdrant (external — already shared)
- In-memory FAISS (rebuilt from Qdrant on each instance startup)
- `document_registry` dict (rebuilt from disk/Qdrant on startup)

Multiple instances can serve read (query) traffic independently. Write (upload) traffic would require distributed locking or a dedicated ingestion service to safely coordinate FAISS and registry updates across instances.

---

## 7. Cost Analysis

### 7.1 Primary Cost Drivers

**OpenAI Embedding API** — called on every document upload and every user question.
- Upload: `n_chunks × 700 words ≈ n_chunks × 525 tokens` per document
- Query: 1 embedding call per question (~15–50 tokens)
- Model: `text-embedding-3-small` — lowest cost embedding tier

**OpenAI Generation API** — called on every answered question.
- Input: system prompt + context (up to 8 chunks × 700 words) + question ≈ 3,000–5,000 tokens
- Output: capped at 500 tokens
- Model: `gpt-4o-mini` — cost-optimized generation model

**Qdrant** — cost depends on plan tier. Write on every chunk upsert. Read on every query.

**Render** — compute cost for the hosted service instance.

### 7.2 Cost Optimization Decisions

- `gpt-4o-mini` selected over `gpt-4o` — significantly lower per-token cost with acceptable quality for document Q&A
- `text-embedding-3-small` selected over `text-embedding-3-large` — 1536 vs 3072 dimensions, lower cost, sufficient accuracy
- `max_tokens=500` hard cap — prevents runaway generation costs
- `temperature=0.1` — low temperature reduces token waste from verbose outputs
- Batch size 32 for embedding — minimizes API round trips per document

---

## 8. Deployment Architecture

```
Client
  │
  │ HTTPS
  ▼
Render (PaaS)
  └─ uvicorn app.main:app --host 0.0.0.0 --port $PORT
        └─ FastAPI Application
              ├─ /storage (ephemeral disk)
              │     ├─ faiss.index
              │     ├─ metadata.json
              │     ├─ document_registry.json
              │     └─ metrics.json
              └─ /logs/app.log (ephemeral)

External Services:
  ├─ Qdrant Cloud    ← vector persistence
  ├─ OpenAI API      ← embeddings + generation
  ├─ Google Gemini   ← generation fallback
  └─ PostHog         ← event tracking
```

**Environment variables required:**
```
OPENAI_API_KEY
GEMINI_API_KEY
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
POSTHOG_API_KEY
POSTHOG_HOST
```

---

## 9. Production Performance

Measured across 25 real queries against 3 live documents (Gemini API docs, FastAPI docs, OpenAI Python SDK docs) on the deployed production system.

| Metric | Value |
|---|---|
| Total queries | 25 |
| Success rate | 100% |
| Grounded answer rate | 100% |
| Answer accuracy | 96% |
| Average latency | 7.16s |
| P50 latency | 6.03s |
| P95 latency | 13.52s |
| P99 latency | 14.08s |
| Refusal rate | 0% |

---

## 10. Configuration Reference

All system parameters are centralized in `app/config.py`. No business constants are hardcoded in logic files.

| Parameter | Value | Description |
|---|---|---|
| `CHUNK_SIZE` | 700 | Words per chunk |
| `CHUNK_OVERLAP` | 120 | Overlap between consecutive chunks |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `TOP_K` | 8 | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | 0.45 | Minimum score to attempt answer |
| `LLM_MODEL` | `gpt-4o-mini` | Primary generation model |
| `LLM_MAX_TOKENS` | 500 | Maximum answer length |
| `TARGET_LATENCY_SECONDS` | 2.0 | Latency target |
| `MAX_FILE_SIZE_MB` | 10 | Upload size limit |
| `MAX_DOCUMENTS_IN_MEMORY` | 100 | Memory safety limit |
| `MAX_CHUNKS_PER_DOCUMENT` | 500 | Chunk count guard |
| `MAX_DOCUMENT_CHARACTERS` | 2,000,000 | Text truncation limit |
| `MAX_HTML_PAGES` | 20 | Web crawler page limit |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |

---


