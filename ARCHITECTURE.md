# System Architecture

## Overview

This document explains the architecture of the AI Document Understanding System. The system is designed with clear layer separation, deterministic behavior, and production-grade error handling.

---

## High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                           │
│  (Streamlit UI / API Clients / curl / Postman)              │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP
┌─────────────────────────────────────────────────────────────┐
│                      API LAYER                              │
│  • FastAPI Routes (app/api/routes.py)                       │
│  • Request Validation (Pydantic models)                     │
│  • Error Handling (HTTP status codes)                       │
│  • Request Logging & Tracing                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    WORKFLOW LAYER                           │
│  • RAG Pipeline (app/workflow/document_qa.py)               │
│  • Similarity Threshold Logic                               │
│  • Refusal Decision Making                                  │
│  • Prompt Construction                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────┴──────────────────┐
         ↓                                     ↓
┌──────────────────────┐            ┌──────────────────────┐
│   MEMORY LAYER       │            │     LLM LAYER        │
│  • Document Loader   │            │  • OpenAI Client     │
│  • Text Chunker      │            │  • Prompt Execution  │
│  • Embedder          │            │  • Response Parsing  │
│  • Vector Store      │            │                      │
│  • Retriever         │            │                      │
└──────────────────────┘            └──────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│                  OBSERVABILITY LAYER                        │
│  • Structured Logging (JSON)                                │
│  • Request Tracing (request_id)                             │
│  • Latency Tracking                                         │
│  • Error Logging                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Request Flow: Upload Document

### 1. **Client → API Layer**
```
POST /upload
Content-Type: multipart/form-data
Body: PDF file
```

### 2. **API Layer Validation**
- Check file extension (`.pdf` only)
- Check file size (≤10MB)
- Generate unique document ID (`doc_abc123...`)
- Log request start with `request_id`

### 3. **Memory Layer Processing**
```python
# app/api/routes.py
text = load_pdf_text("temp.pdf")          # Extract text
chunks = chunk_text(text)                  # Split into 500-word chunks
embeddings = embedder.embed(chunks)        # Generate vectors
vector_store.add(embeddings, chunks, doc_id)  # Store with doc_id tag
```

### 4. **Document Registry Update**
```python
document_registry[doc_id] = {
    "filename": "contract.pdf",
    "chunks_count": 47,
    "upload_timestamp": "2024-01-30T10:15:00Z"
}
```

### 5. **Response**
```json
{
    "document_id": "doc_abc123",
    "filename": "contract.pdf",
    "chunks_created": 47,
    "message": "Document uploaded and indexed successfully"
}
```

---

## Request Flow: Ask Question

### 1. **Client → API Layer**
```
POST /ask
Content-Type: application/json
{
    "session_id": "doc_abc123",
    "question": "What is the contract effective date?"
}
```

### 2. **API Layer Validation**
- Validate question length (10-1000 chars)
- Check document exists
- Generate `request_id` for tracing
- Log request start

### 3. **Workflow Layer: Retrieval**
```python
# app/workflow/document_qa.py
chunks = retrieve_fn(question, top_k=5)
# Returns: [
#   {"text": "...", "similarity_score": 0.87, "doc_id": "doc_abc123"},
#   {"text": "...", "similarity_score": 0.72, ...},
#   ...
# ]
```

### 4. **Workflow Layer: Similarity Threshold Check**
```python
SIMILARITY_THRESHOLD = 0.65

if not chunks or chunks[0]["similarity_score"] < SIMILARITY_THRESHOLD:
    # REFUSE - Low confidence
    return {
        "answer": "I don't have enough confident information...",
        "refused": True,
        "confidence_score": chunks[0]["similarity_score"],
        "reasoning": "Below threshold"
    }
```

### 5. **Workflow Layer: Prompt Construction** (if passing threshold)
```python
prompt = f"""
You are a document understanding assistant.
Use ONLY the context below.

Context:
[Context 1, Confidence: 0.87]
The contract effective date is January 15, 2024...

[Context 2, Confidence: 0.72]
This agreement commences on the effective date...

Question: What is the contract effective date?
Answer:
"""
```

### 6. **LLM Layer: Generation**
```python
# app/llm/client.py
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2
)
answer = response.choices[0].message.content
```

### 7. **Response**
```json
{
    "answer": "The contract effective date is January 15, 2024.",
    "session_id": "doc_abc123",
    "confidence_score": 0.87,
    "refused": false,
    "sources_used": 5,
    "reasoning": null
}
```

---

## Layer Responsibilities

### **API Layer** (`app/api/routes.py`)
**Job:** Thin orchestration, validation, error handling

**Does:**
-  Validates all inputs (Pydantic models)
-  Checks file types and sizes
-  Manages document registry
-  Returns proper HTTP status codes
-  Logs requests and errors

**Does NOT:**
-  Make refusal decisions (that's Workflow layer)
-  Know about embeddings or similarity (that's Memory layer)
-  Call LLM directly (that's LLM layer via Workflow)

---

### **Memory Layer** (`app/memory/`)
**Job:** Document storage, chunking, retrieval

**Components:**
- `loader.py` - Extract text from PDFs
- `chunker.py` - Split text into 500-word chunks with 100-word overlap
- `embedder.py` - Generate embeddings (all-MiniLM-L6-v2, 384 dims)
- `store.py` - FAISS vector index + metadata storage
- `retriever.py` - Similarity search with document filtering

**Does:**
-  Stores multiple documents with isolation (doc_id tagging)
-  Returns similarity scores (for threshold logic)
-  Filters retrieval by document ID

**Does NOT:**
-  Make decisions about refusal (that's Workflow)
-  Generate answers (that's LLM layer)

---

### **Workflow Layer** (`app/workflow/document_qa.py`)
**Job:** RAG orchestration, refusal logic, prompt building

**Critical Logic:**
```python
# THIS IS THE CORE OF THE SYSTEM
if chunks[0]["similarity_score"] < 0.65:
    return REFUSE
else:
    prompt = build_prompt(question, chunks)
    answer = llm_client.generate(prompt)
    return answer
```

**Does:**
-  Controls the entire RAG pipeline
-  Makes refusal decisions (similarity threshold)
-  Builds prompts with anti-hallucination instructions
-  Handles LLM failures gracefully

**Does NOT:**
-  Validate HTTP requests (that's API layer)
-  Store embeddings (that's Memory layer)
-  Call OpenAI directly (that's LLM layer)

---

### **LLM Layer** (`app/llm/client.py`)
**Job:** Execute prompts, return answers

**Does:**
-  Calls OpenAI API
-  Handles API failures
-  Parses responses

**Does NOT:**
-  Decide what to retrieve (that's Workflow + Memory)
-  Decide when to refuse (that's Workflow)
-  Build prompts (that's Workflow)

---

### **Observability Layer** (`app/observability/logger.py`)
**Job:** Structured logging, request tracing

**Does:**
-  JSON-formatted logs
-  Request ID tracking (trace requests end-to-end)
-  Latency measurement
-  Error logging with stack traces

**Output Example:**
```json
{
    "timestamp": "2024-01-30T10:15:23.456Z",
    "level": "INFO",
    "message": "query_completed",
    "request_id": "req_abc123",
    "document_id": "doc_xyz789",
    "latency_seconds": 1.234,
    "confidence_score": 0.87,
    "refused": false
}
```

---

## Multi-Document Architecture

### **Document Isolation**
Each document is stored with metadata:
```python
{
    "doc_abc123": {
        "filename": "contract.pdf",
        "chunks_count": 47,
        "upload_timestamp": "..."
    },
    "doc_xyz789": {
        "filename": "invoice.pdf",
        "chunks_count": 12,
        "upload_timestamp": "..."
    }
}
```

### **Vector Store Structure**
```python
VectorStore:
    FAISS Index: [embedding_1, embedding_2, ..., embedding_N]
    Chunks: [
        {"text": "...", "doc_id": "doc_abc123", "chunk_idx": 0},
        {"text": "...", "doc_id": "doc_abc123", "chunk_idx": 1},
        {"text": "...", "doc_id": "doc_xyz789", "chunk_idx": 0},
        ...
    ]
```

### **Query Filtering**
```python
# Only retrieve from specified document
results = vector_store.query(
    embedding=query_embedding,
    top_k=5,
    doc_id="doc_abc123"  # Filter by document
)
```

---

## Why This Design?

### **1. Separation of Concerns**
- Each layer has ONE job
- Easy to test each layer independently
- Can replace components without breaking others
- Example: Swap FAISS for ChromaDB → only touch Memory layer

### **2. Testability**
- Pure functions for chunking, prompt building
- Classes for stateful systems (embedder, vector store, LLM client)
- Mock any layer in tests
- 57 test cases validate all boundaries

### **3. Observability**
- Every request has unique ID
- Can trace failures through logs
- Latency measured at every step
- Refusal reasons logged

### **4. Failure Isolation**
- LLM failure → Workflow returns error, API stays up
- Bad PDF → Upload fails, other docs unaffected
- Vector store full → New uploads fail, queries still work

### **5. Scalability Path**
Current (v1):
- In-memory FAISS
- Single instance
- Good for 100 documents

Future (v2+):
- Swap FAISS → ChromaDB (persistent)
- Add Redis caching
- Horizontal scaling with shared DB

**Architecture supports evolution without rewrite.**

---

## Configuration Management

All tunable parameters in `app/config.py`:
- `CHUNK_SIZE = 500` (words per chunk)
- `CHUNK_OVERLAP = 100` (overlap for context)
- `SIMILARITY_THRESHOLD = 0.65` (refusal threshold)
- `TOP_K = 5` (retrieval depth)
- `LLM_MODEL = "gpt-4o-mini"` (model selection)

**Why centralized?**
- Change behavior without code changes
- Document design decisions in one place
- Easy A/B testing (change threshold, measure refusal rate)

---

## Key Design Decisions

See `DESIGN_DECISIONS.md` for detailed trade-off analysis.

**Quick summary:**
1. **In-memory FAISS** → Fast, simple (acceptable for v1 scope)
2. **Similarity threshold 0.65** → Empirically tuned (15% refusal, 0% hallucination)
3. **Chunk size 500 words** → Balance between precision and context
4. **Top-k = 5** → Cost/quality sweet spot
5. **Single-instance** → Predictable, easier to debug

---

## Security Considerations

**Current (v1):**
-  No authentication
-  No rate limiting
-  No input sanitization beyond validation
-  File size limits (prevent DoS)
-  File type validation

**Future (Production):**
- Add API key authentication
- Add rate limiting (per user, per doc)
- Add input sanitization
- Encrypt documents at rest
- Audit logging

---

## Performance Characteristics

**Latency (target: 1-2 seconds per query):**
- Embedding generation: ~100-200ms
- FAISS search: ~10-50ms (depends on index size)
- LLM generation: ~800-1500ms (gpt-4o-mini)
- **Total: ~1-2 seconds** 

**Memory:**
- ~384 floats per chunk (embeddings)
- ~500 words per chunk (text)
- **100 docs × 50 chunks/doc = 5000 chunks**
- **~50MB for 100 documents** (acceptable for single instance)

**Cost (OpenAI):**
- gpt-4o-mini: ~$0.15 per 1M input tokens
- Average query: ~2000 tokens (5 chunks × 400 tokens/chunk)
- **~$0.0003 per query** (very cheap)

---
