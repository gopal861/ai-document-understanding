# AI Document Understanding System — Design Document

**Author:** Gopal Khandare
**Date:** February 2026
**Status:** Production
**Version:** 1.0
**Deployments:**
- Render: https://ai-document-understanding.onrender.com
- Hugging Face: https://gopal861-ai-document-understanding.hf.space

---

## 1. Design Goals

### 1.1 Ground Every Answer in Document Context

The core problem with LLMs in document Q&A is hallucination — the model generates a confident answer even when the document does not contain the answer. This system is designed so that every answer is traceable to a retrieved document chunk with a measurable confidence score.

**Goal:** Zero out-of-context answers reaching the user.

**Mechanism:** Similarity threshold gate. If retrieved chunks score below `SIMILARITY_THRESHOLD = 0.45`, the system refuses to generate an answer entirely.

**Why we chose this approach:** Post-retrieval confidence gating is more reliable than asking the LLM to self-refuse. LLMs under-refuse — they are trained to be helpful and will generate an answer rather than admit insufficient context. By enforcing the gate at the retrieval layer before the LLM is ever called, the decision is deterministic and costs zero tokens when refusing. A threshold of 0.45 cosine similarity was chosen after evaluation — high enough to filter genuinely irrelevant retrievals, low enough to avoid false refusals when relevant content exists across multiple chunks.

**Measured result:** 100% grounded answer rate across 25 real queries on a live deployed system.

---

### 1.2 Support Multiple Document Sources

Users do not only have PDFs. They work with GitHub repositories, API documentation sites, markdown files, and JS-rendered pages.

**Goal:** Ingest any document source without requiring user pre-processing.

**Mechanism:** Unified `loader.py` with source-routing logic — PDF via `pypdf`, static HTML via `BeautifulSoup` recursive crawler, JS-rendered pages via `Playwright` fallback, GitHub repos via GitHub API tree traversal, raw markdown via HTTP GET.

**Why we chose a unified loader:** A single `load_text(source)` interface means the ingestion pipeline above it — chunker, embedder, vector store — never changes regardless of source type. Adding a new source type requires only adding a branch inside `loader.py`, not touching any other component. The design contract enforces that all loader paths are fully synchronous, which makes executor-based async wrapping straightforward and prevents event loop blocking in FastAPI.

**Why Playwright as fallback instead of primary:** Playwright launches a full Chromium browser, which adds 2–4 seconds of overhead per page. Static HTML crawling with `BeautifulSoup` is fast and sufficient for most documentation sites. Playwright is triggered only when the static result is under 1500 characters — a reliable signal that the page requires JavaScript rendering.

---

### 1.3 Survive Provider Outages

A single LLM provider dependency is a single point of failure. Production systems cannot assume external APIs are always available.

**Goal:** Continue answering questions even if the primary LLM provider is unavailable.

**Mechanism:** `MultiModelLLMClient` with ordered fallback. OpenAI `gpt-4o-mini` is primary. Google Gemini `gemini-2.0-flash` is secondary. Fallback is automatic and transparent to the caller.

**Why we chose OpenAI as primary and Gemini as fallback:** GPT-4o-mini offers the best balance of cost and quality for factual document Q&A. Gemini 2.0 Flash is a capable, fast, and independently operated model — it runs on Google's infrastructure, making it statistically unlikely to fail simultaneously with OpenAI. Two providers from two different companies with separate infrastructure is a meaningful reliability improvement over single-provider design.

**Why we validate Gemini at startup:** A test call to Gemini on initialization confirms it is reachable and responding before it is ever needed as a fallback. A provider that appears configured but is unreachable would cause hidden failures during actual outages — exactly when reliability matters most.

---

### 1.4 Survive Restarts Without Data Loss

Deployed on Render's free tier, which restarts instances on redeploy and on inactivity. In-memory state is wiped on every restart.

**Goal:** Full document corpus survives restarts and redeploys without requiring users to re-upload.

**Mechanism:** Dual storage — FAISS index and metadata written to local disk as a warm cache. Qdrant cloud is the persistent ground truth. On startup, if local state is empty, `VectorStore` rebuilds everything from Qdrant via paginated scroll.

**Why dual storage instead of Qdrant alone:** Qdrant is the durable layer, but rebuilding from Qdrant on every query would add cloud round-trip latency to every operation. Local FAISS and JSON files serve as a fast warm cache that makes the system fully operational after startup without waiting for a complete Qdrant rebuild on the first request.

**Why Qdrant over a self-hosted vector database:** Qdrant Cloud provides persistence, cosine similarity search, payload filtering, and pagination out of the box. Self-hosting would require managing infrastructure that adds no value for this system's scale. The Qdrant free tier is sufficient for the current document corpus and the cloud-managed service eliminates operational overhead.

---

### 1.5 Full Observability

Every request must be diagnosable without requiring external tooling or log aggregation infrastructure.

**Goal:** Any production issue can be diagnosed from logs, metrics, or event data alone.

**Mechanism:** Three independent observability layers — JSON structured logs to console and file, in-process metrics tracker with full latency history, PostHog event tracking for all key system events. Every request receives a UUID injected at middleware level and carried through all three layers.

**Why three layers instead of one:** Each layer serves a different diagnostic purpose. JSON logs give line-by-line execution trace. Metrics give aggregate performance view (P50, P95, error rates). PostHog gives product-level event history — which documents were uploaded, which questions were asked, which calls failed. They operate independently — a PostHog failure never affects logging, and a logging failure never affects metrics.

**Why request-level UUID tracing:** A single `request_id` links every log line, metric record, and PostHog event from a single request across all three layers. When debugging a specific failure, filtering by `request_id` gives the complete execution trace across the entire system in one query.

---

## 2. System Constraints

### 2.1 Memory Constraint

**Problem:** Embedding 500 chunks per document at 1536 dimensions each in RAM accumulates quickly at scale.

**Constraints enforced:**
- `MAX_CHUNKS_PER_DOCUMENT = 500`
- `MAX_DOCUMENTS_IN_MEMORY = 100`
- `MAX_DOCUMENT_CHARACTERS = 2,000,000`

**Why these specific limits:** At 1536 float32 dimensions per vector, 500 chunks per document occupies approximately 3MB of FAISS index space per document. 100 documents occupies ~300MB, which fits within Render's instance memory allocation with headroom for the application itself. The character limit is a upstream guard that prevents the chunker from ever being handed text large enough to violate the chunk count limit.

**Why enforce at multiple layers:** `MAX_DOCUMENT_CHARACTERS` is enforced in both `loader.py` and `chunker.py` independently. This is defense-in-depth — if a loader path fails to truncate, the chunker catches it before embedding is attempted.

---

### 2.2 Ingestion Safety Constraint

**Problem:** Without limits, web crawlers consume unbounded memory, time, and API quota.

**Constraints enforced:**
- `MAX_HTML_PAGES = 20` pages per crawl
- `MAX_DOCUMENT_CHARACTERS = 2,000,000` at loader and chunker
- URL domain allowlist enforced in `validate_url()` before any HTTP request
- Ingestion retries capped at `INGESTION_MAX_RETRIES = 3` with 2-second delay

**Why a domain allowlist rather than open URL ingestion:** Unrestricted URL ingestion creates a server-side request forgery (SSRF) surface. An attacker could point the ingestion endpoint at internal network resources, metadata endpoints, or arbitrary external services. The allowlist limits ingestion to known, legitimate documentation domains and eliminates this attack surface entirely.

**Why retry with delay instead of immediate failure:** Network timeouts during document fetching are transient. A documentation site may return a 503 temporarily, or a PDF download may time out on a slow connection. Three retries with 2-second delays handle the majority of transient failures without user involvement, while the cap prevents indefinite hanging on genuinely unavailable sources.

---

### 2.3 Latency Constraint

**Problem:** Each query pipeline involves an embedding API call, a vector search, and an LLM generation call. Each step adds latency.

**Measured performance on live system:**
- P50: 6.03s
- Average: 7.16s
- P95: 13.52s

**Where latency comes from:** The dominant cost is two sequential external API calls — OpenAI embedding (~300–800ms) and OpenAI generation (~2–6s depending on answer length). These are network-bound calls to external services and represent irreducible latency in the current architecture. Render's deployment region and instance warm-up account for the tail latency seen at P95.

**Optimization in place:** Embeddings are generated in batches of 32 during ingestion, minimizing API round trips per document. Query embedding is a single API call on one text string — already minimal.

**Why this is acceptable for the current deployment:** This system is a document Q&A API, not a real-time chat interface. A 6-second response time is within acceptable range for deep document retrieval tasks where accuracy is the primary requirement. The architecture is designed so that future optimizations — query embedding caching, async Qdrant operations, streaming responses — can be added without changing the core pipeline structure.

---

### 2.4 Concurrency Constraint

**Problem:** Multiple simultaneous uploads both call `vector_store.add()` and attempt to write to the shared FAISS index. FAISS is not thread-safe for concurrent writes.

**Constraint:** `ingestion_lock = threading.Lock()` wraps all `vector_store.add()` calls during upload.

**Why threading.Lock() instead of asyncio.Lock():** The FAISS write operation is synchronous. It runs inside an executor (not the event loop) and must be protected by a threading primitive, not an asyncio primitive. An asyncio lock would not protect code running in a thread pool executor.

**Why only the FAISS write is locked:** Qdrant upserts — which happen before the lock is acquired — are natively thread-safe and can run concurrently. Serializing only the in-memory FAISS write minimizes the time spent holding the lock. Qdrant operations, which are the slower network calls, are never serialized.

---

### 2.5 URL Security Constraint

**Problem:** Open URL ingestion creates a server-side request forgery surface and allows ingestion of arbitrary external content.

**Constraint:** `ALLOWED_DOMAINS` allowlist in `config.py`. Any URL outside this list is rejected with `ValueError` before any HTTP request is made.

```
github.com                raw.githubusercontent.com
platform.openai.com       ai.google.dev
docs.python.org           fastapi.tiangolo.com
huggingface.co            docs.anthropic.com
```

**Why validation before the HTTP request, not after:** Validating the domain before making any network call is the correct approach. Allowing the request to proceed and then discarding the result still exposes internal network addresses, timing information, and server resources to the caller.

---

## 3. Architecture

### 3.1 Layer Structure

```
┌─────────────────────────────────────────────┐
│             HTTP Layer (main.py)             │
│   CORS | UUID Middleware | Exception Handler │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│           API Layer (routes.py)              │
│   Upload | Ask | List | Delete | Health      │
└────┬───────────────┬────────────────────────┘
     │               │
┌────▼────┐   ┌──────▼──────────────────────┐
│Ingestion│   │    Query Workflow             │
│Pipeline │   │    (document_qa.py)          │
└────┬────┘   └──────┬──────────────────────┘
     │               │
┌────▼───────────────▼────────────────────────┐
│           Memory Layer                       │
│  loader → chunker → embedder → store        │
│              retriever ←──────────┘         │
└────────────────────┬────────────────────────┘
                     │
     ┌───────────────┴───────────────┐
     │                               │
┌────▼──────────┐         ┌──────────▼──────┐
│ FAISS (local) │         │ Qdrant (cloud)  │
│ warm cache    │         │ persistent      │
└───────────────┘         └─────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│           LLM Layer                          │
│   OpenAI gpt-4o-mini (primary)              │
│   Gemini gemini-2.0-flash (fallback)        │
└─────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│         Observability Layer                  │
│  JSON Logger | MetricsTracker | PostHog     │
└─────────────────────────────────────────────┘
```

**Why strict layering:** Each layer has a single defined responsibility and communicates only with its immediate neighbor. This means changing the LLM provider requires only changing `multi_model_client.py`. Changing the vector store requires only changing `store.py`. No other layer needs to be aware of implementation details below it. This is the structural reason the system can support dual LLM providers and dual storage backends without the API layer or workflow layer knowing which is in use.

---

### 3.2 Ingestion Pipeline

```
POST /upload (file or URL)
        │
        ▼
routes.py: generate doc_id (uuid4 hex, 12 chars)
        │
        ▼
load_text_with_retry() — async executor wrapper (3 retries, 2s delay)
        │
        ├── PDF → load_pdf_text() → pypdf PdfReader
        ├── GitHub URL → load_github_repo_markdown() → GitHub API
        ├── raw.githubusercontent.com → load_raw_markdown_url()
        ├── HTML (static) → load_html_docs_recursive()
        └── HTML (JS-rendered) → load_dynamic_html_playwright()
                │
                ▼ (all paths enforce MAX_DOCUMENT_CHARACTERS)
        chunk_text(text, size=700, overlap=120)
                │
                ▼ (word-based sliding window, step=580)
        embedder.embed(chunks, batch_size=32)
                │
                ▼ (OpenAI text-embedding-3-small, 1536-dim, L2 normalized)
        with ingestion_lock:
            vector_store.add(embeddings, chunks, doc_id)
                │
                ├── Qdrant: upsert PointStruct (vector + payload)
                ├── FAISS: index.add(embeddings)
                └── metadata.json + faiss.index written to disk
        │
        ▼
document_registry[doc_id] = {filename, chunks_count, timestamp}
save_document_registry() → storage/document_registry.json
        │
        ▼
PostHog: track_document_upload(doc_id, filename, chunks, latency)
        │
        ▼
UploadResponse(document_id, filename, chunks_created)
```

---

### 3.3 Query Pipeline

```
POST /ask (document_id, question)
        │
        ▼
routes.py: validate document_id in document_registry
        │
        ▼
document_qa.answer_question()
        │
        ▼ Step 1: Retrieve
retrieve_fn(question, top_k=8)
        │
        ├── embedder.embed([question]) → 1×1536 normalized vector
        └── store.query(embedding, top_k=8, doc_id=filter)
                │
                └── Qdrant search with doc_id payload filter
                    Returns: [{text, doc_id, chunk_idx, similarity_score}]
        │
        ▼ Step 2: Validate
if not context_chunks:
    return refused=True, confidence=0.0
        │
if top_score < SIMILARITY_THRESHOLD (0.45):
    return refused=True, confidence=top_score
        │
        ▼ Step 3: Build Prompt
build_document_prompt(question, context_chunks, model_type="cloud")
        │
        ├── DOCUMENT_QA_SYSTEM_PROMPT (from system_prompts.py)
        ├── Context block: each chunk labeled [Context N | Confidence: X.XXX]
        └── Refusal instruction if answer not in context
        │
        ▼ Step 4: Generate
llm_client.generate(prompt)
        │
        ├── OpenAI (primary): gpt-4o-mini, temp=0.1, max_tokens=500
        └── Gemini (fallback): gemini-2.0-flash
        │
        ▼ Step 5: Track + Return
PostHog: track llm_completed event
        │
AskResponse(answer, document_id, confidence_score, refused, sources_used)
```

---

## 4. Component Design

### 4.1 Chunker (`chunker.py`)

**Pattern:** Sliding window over word tokens.

**Parameters:**
- `CHUNK_SIZE = 700` words
- `CHUNK_OVERLAP = 120` words
- Step size = `700 - 120 = 580` words

**Why word-based chunking instead of token-based:** Token-based chunking requires the tiktoken library and is tightly coupled to a specific model's tokenization scheme. If the embedding model changes, token counts change. Word-based chunking is model-agnostic, requires no additional dependencies, and produces consistent results regardless of which embedding model is in use. A word count of 700 words corresponds to approximately 900–1000 tokens for typical English technical documentation — well within `text-embedding-3-small`'s 8191-token context limit.

**Why 120-word overlap:** Overlap prevents information loss at chunk boundaries. A key sentence that falls at the end of one chunk and the beginning of the next would be retrievable from either chunk. 120 words (~17% of chunk size) is enough to preserve cross-boundary context without significantly increasing the number of chunks.

**Safety guarantees:** Empty text rejected early. Overlap validated to be less than chunk size. Infinite loop prevented by `step <= 0` guard. Text truncated at `MAX_DOCUMENT_CHARACTERS` before chunking begins.

---

### 4.2 Embedder (`embedder.py`)

**Model:** `text-embedding-3-small` — 1536 dimensions.

**Why `text-embedding-3-small` over alternatives:**
- Over `text-embedding-ada-002`: Newer generation with significantly better retrieval quality at similar cost.
- Over `text-embedding-3-large` (3072 dims): Doubles storage and memory cost with marginal quality improvement for document Q&A tasks. 1536 dimensions provides sufficient representational capacity for technical documentation.
- Over local embedding models: Eliminates GPU dependency, simplifies deployment on CPU-only Render instances, and removes model versioning complexity.

**Why batch size 32:** OpenAI's embedding API accepts up to 2048 inputs per call but large batches increase memory pressure when building the embedding array. Batches of 32 balance API efficiency (fewer round trips) with predictable memory usage per batch.

**Why L2 normalization:** Normalized vectors make cosine similarity equivalent to dot product. FAISS `IndexFlatIP` computes inner product — normalizing vectors means this inner product equals cosine similarity, enabling consistent similarity scoring whether the query comes from FAISS or Qdrant (which uses cosine distance natively).

**Dimension mapping (enforced at init):**
- `text-embedding-3-small` → 1536
- `text-embedding-3-large` → 3072
- Any other model → `ValueError` at startup, system refuses to start with misconfigured model

---

### 4.3 Vector Store (`store.py`)

**Dual storage architecture:**

| Layer | Technology | Role |
|---|---|---|
| Cloud persistence | Qdrant | Durable store. All writes go here. Survives restarts. |
| Local cache | FAISS IndexFlatIP | In-memory index rebuilt from Qdrant on startup. |
| Chunk metadata | `storage/metadata.json` | Maps chunk index to text and doc_id. |
| Document registry | `storage/document_registry.json` | Maps doc_id to filename and chunk count. |

**Why FAISS + Qdrant together instead of Qdrant alone:**
Qdrant is the source of truth for persistence, but every query going to Qdrant adds a cloud network round trip. FAISS in-memory provides sub-millisecond local search. The dual design means the system can serve queries from local FAISS cache after startup while Qdrant handles all persistence guarantees. FAISS is rebuilt from Qdrant on startup, so both are always in sync.

**Why `IndexFlatIP` for FAISS:** Flat index performs exact nearest-neighbor search — no approximation. For the document counts this system handles (up to 50,000 vectors at maximum config), exact search is fast enough and guarantees no recall loss from approximation. Approximate methods like HNSW or IVF would add index-building complexity with no meaningful latency benefit at this scale.

**Startup rebuild sequence:**
```
VectorStore.__init__()
    → _load_from_disk()           (attempt local FAISS + metadata)
    → if len(_chunks) == 0:
        _rebuild_from_qdrant()    (paginated scroll, 100 points/page)
                                  (rebuilds _chunks, _doc_chunk_count, FAISS)
        _save_to_disk()           (restore local cache for next startup)
```

---

### 4.4 Loader (`loader.py`)

**Design contract:** Fully synchronous. `asyncio` is explicitly forbidden inside this file. Async execution is handled at the routes layer via `loop.run_in_executor(None, load_text, source)`.

**Why enforced synchronous design:** Playwright's synchronous API (`sync_playwright`) cannot be called from within a running asyncio event loop. Keeping the entire loader synchronous means all paths — including the Playwright fallback — can safely run in an executor without per-path async/sync management. This simplifies testing, debugging, and future extension of the loader.

**Source routing logic:**
```
source starts with http/https:
    domain == raw.githubusercontent.com → load_raw_markdown_url()
    domain == github.com               → load_github_repo_markdown()
    else:
        try static HTML crawl (BeautifulSoup, recursive, max 20 pages)
        if result < 1500 chars:
            try Playwright dynamic fallback
            use whichever result is longer

source ends with .pdf → load_pdf_text()
source ends with .md  → load_markdown_text()
else → ValueError (unsupported source)
```

**Playwright configuration:** Chromium headless with `--no-sandbox`, `--disable-dev-shm-usage`, `--disable-gpu`. Browser path set to `/opt/render/project/.playwright` for Render compatibility. These flags are required for headless Chromium in containerized environments without a display server.

---

### 4.5 LLM Client (`multi_model_client.py`)

**Why this component exists as a separate class:** Encapsulating LLM provider logic behind a single `generate(prompt) → str` interface means the workflow layer (`document_qa.py`) never knows or cares which provider served the request. Provider initialization, health checking, fallback logic, latency tracking, and error handling are all contained here.

**Initialization:** Both providers initialized at startup. OpenAI validated by API key prefix check. Gemini validated with a live test call (`"ping"`). Provider availability flags set independently.

**Fallback logic:**
```
generate(prompt):
    if openai_available:
        try _generate_openai(prompt)
        success → return result
        failure → log warning, fall through

    if gemini_available:
        try _generate_gemini(prompt)
        success → return result
        failure → log warning

    raise RuntimeError("No LLM backend available")
```

**Why ordered fallback instead of concurrent calls:** Making concurrent calls to both providers and using whichever responds first would halve latency but double cost on every successful request. The ordered approach only calls Gemini when OpenAI fails, keeping Gemini cost at zero during normal operation.

**OpenAI call:** `gpt-4o-mini`, `temperature=0.1`, `max_tokens=500`. System prompt injected as a separate `role: system` message — the correct pattern for the OpenAI Chat Completions API.

**Gemini call:** `gemini-2.0-flash`. System prompt prepended to user content as a single string. The Google GenAI SDK used in this integration (`genai.Client`) does not accept a separate system role parameter in `generate_content`, so system prompt is prepended to the user content.

**Latency tracking:** Every provider call wrapped in `_timed_call()` which records `provider` and `latency_seconds` to structured logs on success.

---

### 4.6 Document QA Workflow (`document_qa.py`)

**Why this orchestration layer exists:** Separating the QA workflow from the API routes means the pipeline — retrieve, validate, prompt, generate — is independently testable and reusable. The workflow receives a `retrieve_fn` callable and an `llm_client` instance as arguments, which allows any retriever and any LLM client to be injected without changing the pipeline itself.

**Orchestration steps (in order):**

1. Call `retrieve_fn(question, top_k=8)` → get context chunks
2. Track retrieval in PostHog
3. If no chunks → refuse, `confidence=0.0`
4. If `top_score < 0.45` → refuse with score and reasoning
5. Build grounded prompt via `build_document_prompt()`
6. Call `llm_client.generate(prompt)`
7. On LLM exception → catch, track error in PostHog, return refused response
8. Track `llm_completed` in PostHog with full latency metrics
9. Return structured response

**Response schema:**
```json
{
  "answer": "string",
  "document_id": "string",
  "confidence_score": 0.847,
  "refused": false,
  "sources_used": 8,
  "reasoning": null
}
```

**Why `reasoning` field is returned:** When the system refuses, `reasoning` contains a human-readable explanation — the similarity score, the threshold, and why the refusal was triggered. This enables API consumers to distinguish between "document does not contain this information" and "retrieval failed" without parsing error messages.

---

### 4.7 Prompt Design (`prompt_builder.py`, `system_prompts.py`)

**Why prompts are centralized in `system_prompts.py`:** Prompts are a core part of system behavior. Distributing them across workflow and LLM client files makes them invisible and hard to audit. A single file that defines all system behavior makes prompt iteration, A/B testing, and review straightforward. The rule is enforced in code — no string prompts exist anywhere except in `system_prompts.py`.

**Why confidence scores are injected into the context block:**
```
[Context 1 | Confidence: 0.823]
<chunk text>

[Context 2 | Confidence: 0.791]
<chunk text>
```
Including confidence scores in the prompt gives the LLM signal about which chunks are most relevant. A chunk scored 0.82 should be weighted more heavily than one scored 0.51. This allows the model to produce better-calibrated answers when context quality varies across retrieved chunks.

**System prompt design decisions:**
- Refuse ONLY if answer truly cannot be found — minimizes false refusals
- Explicitly permit combining information across multiple chunks — critical for answers that require synthesis
- Do not mention context or chunks in the answer — answers read as direct responses, not as "according to the provided context..."
- Priority order: Accuracy → Grounded reasoning → Completeness → Conciseness

**Why a defined refusal phrase:** The exact phrase `"I don't have enough information in the document to answer this."` is specified in the system prompt. This makes programmatic refusal detection possible on the client side — the `refused` flag in the response already signals this, but the consistent phrase means clients can also detect it from answer text if needed.

---

### 4.8 Observability Stack

**Three independent layers — each operates without depending on the others:**

**Layer 1 — JSON Logger (`logger.py`)**

**Why JSON over plaintext logs:** JSON logs are machine-parseable. Every field — timestamp, level, request_id, module, latency — is a structured key-value pair. This means log aggregation tools (Datadog, CloudWatch, Loki) can index and query them without parsing. Plaintext logs require regex parsing that is fragile and format-dependent.

- All logs structured as JSON with timestamp, level, logger, message, module, function, line, custom extras
- Dual output: stdout + `logs/app.log`
- Reserved field protection: custom extra fields that collide with standard `LogRecord` attributes are prefixed with `extra_` automatically
- Noisy library suppression: `urllib3`, `httpx`, `openai` set to WARNING

**Layer 2 — Metrics Tracker (`metrics.py`)**

**Why in-process metrics instead of a metrics sidecar:** For a single-instance deployment on Render, an in-process metrics tracker has zero infrastructure cost and is immediately available at `GET /metrics`. Adding Prometheus, StatsD, or a metrics agent would require additional infrastructure with no benefit at current scale.

- Thread-safe via `threading.Lock()` on all writes
- Tracks: total/success/failed requests, total latency, avg latency, full latency history list
- Full latency history enables any-percentile calculation via `get_latency_percentile(p)`
- Persisted to `storage/metrics.json` on every update — survives process restart
- Exposed at `GET /metrics`

**Layer 3 — PostHog (`posthog_client.py`)**

**Why PostHog for event tracking:** PostHog provides a product-analytics event stream with a generous free tier. It gives a time-series view of system usage — how many documents uploaded, which endpoints are called, where errors occur — that complements the aggregate metrics from Layer 2. PostHog is also designed for user identity tracking, making it ready to extend into user-level analytics when authentication is added.

- 5 event types tracked: `document_uploaded`, `question_asked`, `retrieval_completed`, `llm_completed`, `system_error`
- All tracking wrapped in try/except — PostHog failure never propagates to the request
- `request_id` (UUID) used as `distinct_id` — links all events from a single request
- `flush_interval=1` for near-real-time dashboard visibility

---

## 5. API Design

### 5.1 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Ingest document (file or URL) |
| `POST` | `/ask` | Ask question against a document |
| `GET` | `/documents` | List all documents with metadata |
| `DELETE` | `/documents/{id}` | Remove document and all associated vectors |
| `GET` | `/health` | System health and vector store stats |
| `GET` | `/metrics` | Request metrics and latency percentiles |
| `GET` | `/` | API information |

**Why FastAPI:** FastAPI provides automatic OpenAPI documentation generation, Pydantic-based request and response validation, native async support, and type-safe handler signatures — all without configuration. Pydantic validation means malformed requests are rejected before they reach any business logic, and the auto-generated `/docs` endpoint provides a live API explorer at zero cost.

### 5.2 Request Middleware

Every HTTP request passes through `log_requests` middleware before reaching any handler:

1. Generate `uuid4` as `request_id`
2. Attach to `request.state.request_id`
3. Call `posthog_client.identify_user(request_id)`
4. Log request start with method, path, client IP
5. Execute handler
6. Record latency in `metrics_tracker`
7. Log completion with status code and latency, or log error on exception

**Why middleware for cross-cutting concerns:** Request ID generation, timing, and logging are not the responsibility of individual route handlers. Middleware guarantees these concerns are applied uniformly to every request regardless of which endpoint is called, without each handler needing to implement them.

---

## 6. Storage Design

### 6.1 Storage Layer Map

| Data | Technology | Location | Durability |
|---|---|---|---|
| Vectors + payloads | Qdrant Cloud | Cloud | Permanent |
| FAISS index | File | `storage/faiss.index` | Instance lifetime |
| Chunk metadata | JSON | `storage/metadata.json` | Instance lifetime |
| Document registry | JSON | `storage/document_registry.json` | Instance lifetime |
| Request metrics | JSON | `storage/metrics.json` | Instance lifetime |
| Application logs | JSON lines | `logs/app.log` | Instance lifetime |

**Key architectural principle:** Everything in `storage/` and `logs/` is ephemeral on Render — it lives only as long as the current instance. Qdrant is the only data layer that survives redeploys. The entire system is designed to rebuild all local state from Qdrant on startup. This means the durable state is always in Qdrant, and local files are performance optimizations, not sources of truth.

---

### 6.2 Qdrant Point Schema

Each vector point stored with:
```json
{
  "id": "<uuid4>",
  "vector": [1536 × float32],
  "payload": {
    "text": "<chunk text content>",
    "doc_id": "doc_abc123def456",
    "chunk_idx": 0
  }
}
```

Collection: `documents`. Distance metric: Cosine. Dimension: 1536.
Auto-created on first startup if not present.

**Why cosine distance:** Cosine similarity measures the angle between vectors, making it insensitive to vector magnitude. Since all embeddings are L2-normalized before storage, cosine similarity and dot product are equivalent — but cosine is the correct semantic choice for comparing text embedding similarity regardless of text length.

**Why doc_id as a payload field instead of a separate collection per document:** A single collection with payload filtering is simpler to manage than hundreds of collections. Qdrant's payload filter is efficient and allows querying across all documents or filtering to one document with the same index.

---

## 7. Hallucination Prevention Design

Three independent gates prevent hallucinated answers from reaching users:

**Gate 1 — Empty Retrieval (pre-LLM)**
```
len(context_chunks) == 0 → refused=True, confidence=0.0
```
If no chunks are retrieved, the question cannot be answered from this document. The LLM is never called. Cost: zero tokens.

**Gate 2 — Similarity Threshold (pre-LLM)**
```
top_score < 0.45 → refused=True, confidence=top_score
```
If the best-matching chunk scores below 0.45 cosine similarity, the question is not answerable with sufficient confidence from the available context. The LLM is never called. Cost: zero tokens.

**Gate 3 — Prompt-Level Grounding (in-LLM)**
```
System prompt: answer ONLY from context.
If answer not in context, return exact refusal phrase.
```
Even if Gates 1 and 2 pass, the LLM is explicitly instructed to self-refuse if the context does not support the answer. This is defense-in-depth — it catches cases where retrieved chunks are superficially similar but don't actually answer the question.

**Why three gates instead of one:** Each gate catches a different failure mode. Gate 1 catches queries with no semantic match at all. Gate 2 catches queries with weak matches that don't meet the confidence bar. Gate 3 catches queries where chunks scored above threshold but don't contain the specific answer. No single gate covers all three failure modes.

**Measured outcome:** 96% answer accuracy, 100% grounded answer rate across 25 production queries.

---

## 8. Failure Modes

### 8.1 OpenAI API Unavailable

**What happens:** `MultiModelLLMClient.generate()` catches the exception on the OpenAI call, logs a warning, and falls through to Gemini automatically.

**User impact:** Slightly higher latency — Gemini is secondary. Answer is still returned. The fallback is transparent to the user.

**Terminal failure condition:** Both OpenAI AND Gemini are unavailable simultaneously → `RuntimeError("No LLM backend available")` → 500 response to user. This is the only condition under which an answered question cannot be served.

---

### 8.2 Qdrant Unavailable

**What happens:** `vector_store.add()` or `vector_store.query()` raises an exception that propagates through the handler.

**User impact:** 500 error on upload or ask for the duration of the Qdrant outage.

**Resilience characteristic:** Qdrant Cloud provides its own availability guarantees. The system has no retry logic on Qdrant calls — it relies on Qdrant's managed infrastructure availability.

---

### 8.3 Instance Restart

**What happens:** All in-memory state (FAISS index, metadata, document registry) is lost.

**Recovery:** On startup, `VectorStore._rebuild_from_qdrant()` paginates through all Qdrant points and reconstructs the FAISS index, chunk metadata, and doc chunk counts. `load_document_registry()` restores the document registry from disk JSON, and `rebuild_registry_from_vector_store()` reconciles any gaps between disk state and Qdrant state.

**Recovery time:** Proportional to the number of vectors in Qdrant. At 100 vectors/second scroll rate and 50,000 maximum vectors, rebuild completes in under 10 minutes in the worst case. Normal operation at current scale completes in seconds.

---

### 8.4 Playwright Unavailable

**What happens:** `load_dynamic_html_playwright()` raises an exception.

**Recovery:** The loader always attempts static HTML crawling first. Playwright is only called when the static result is under 1500 characters. If Playwright fails, the exception propagates to `load_text_with_retry()` in routes, which retries up to 3 times. If all retries fail, the upload returns a 500 error. The static HTML result from the first pass is not used as a partial fallback — either Playwright succeeds and returns useful content, or the upload fails cleanly.

---

### 8.5 Embedding API Failure

**What happens:** `embedder.embed()` raises `RuntimeError`. The exception propagates through `upload_document()`.

**User impact:** Upload fails with a 500 error. PostHog error event is tracked. No partial state is written — Qdrant and FAISS are only written to after embedding succeeds. The system remains in a consistent state.

---

## 9. Configuration Reference

All tunable parameters in `app/config.py`. No business constants are hardcoded in logic files.

| Parameter | Value | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 700 | Words per chunk |
| `CHUNK_OVERLAP` | 120 | Overlap between consecutive chunks |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `TOP_K` | 8 | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | 0.45 | Minimum cosine score to attempt answer |
| `LLM_MODEL` | `gpt-4o-mini` | Primary generation model |
| `LLM_TEMPERATURE` | 0.1 | Low temperature for factual answers |
| `LLM_MAX_TOKENS` | 500 | Max answer length |
| `MAX_FILE_SIZE_MB` | 10 | Upload size limit |
| `MAX_DOCUMENTS_IN_MEMORY` | 100 | Safety limit |
| `MAX_CHUNKS_PER_DOCUMENT` | 500 | Memory guard |
| `MAX_DOCUMENT_CHARACTERS` | 2,000,000 | Text truncation limit |
| `MAX_HTML_PAGES` | 20 | Crawler page limit |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |

---

## 10. Deployment

**Platforms:** Render (PaaS) · Hugging Face Spaces

| Platform | URL |
|---|---|
| Render | https://ai-document-understanding.onrender.com |
| Hugging Face | https://gopal861-ai-document-understanding.hf.space |

**Process:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Runtime:** Python 3.11.9

**Key dependencies:**

| Package | Version | Role |
|---|---|---|
| fastapi | 0.129.0 | API framework |
| uvicorn | 0.41.0 | ASGI server |
| pydantic | 2.12.5 | Request/response validation |
| openai | 2.21.0 | Embedding + generation |
| google-genai | 0.3.0 | Gemini fallback |
| faiss-cpu | 1.13.2 | In-memory vector index |
| qdrant-client | 1.9.1 | Persistent vector store |
| pypdf | 6.7.0 | PDF text extraction |
| beautifulsoup4 | 4.14.3 | HTML parsing |
| playwright | 1.46.0 | JS-rendered page extraction |
| posthog | 3.5.0 | Event tracking |
| python-dotenv | 1.2.1 | Environment config |

---

## 11. Design Patterns

**Layered Architecture:** HTTP → API → Workflow → Memory → LLM → Observability. Each layer has a defined contract and communicates only with its immediate neighbor. No layer skips another.

**Singleton Components:** `embedder`, `vector_store`, `llm_client` initialized once at module load in `routes.py` and shared across all requests. This avoids per-request model initialization cost and ensures a single FAISS index is shared across all concurrent requests.

**Dependency Injection:** `document_qa.answer_question()` receives `retrieve_fn` and `llm_client` as arguments rather than importing them directly. This decouples the workflow from specific implementations and makes unit testing straightforward — any callable can be injected as the retriever or LLM.

**Retry with Backoff:** Document loading uses `INGESTION_MAX_RETRIES = 3` with `INGESTION_RETRY_DELAY = 2s` between attempts. Handles transient network failures without user involvement.

**Fail-Safe Observability:** PostHog and metrics calls are wrapped in try/except. Observability failures never propagate to the request that triggered them. The system prefers partial observability over request failure.

**Configuration Centralization:** Zero business constants hardcoded in logic files. All values imported from `config.py`. This means tuning retrieval depth, chunk size, or similarity threshold requires changing one file with no risk of missing a hardcoded value elsewhere.

**Executor Isolation:** All synchronous I/O — file loading, Playwright, web requests — runs in `loop.run_in_executor(None, ...)` to avoid blocking the FastAPI event loop. This is the correct pattern for mixing sync and async code in FastAPI without degrading concurrent request handling.

---

## 12. Production Performance

Measured on the live deployed system across 25 real queries against 3 documents — Gemini API documentation, FastAPI documentation, and OpenAI Python SDK documentation.

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

