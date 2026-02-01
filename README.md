# AI Document Understanding System

## Overview

Production-grade Generative AI system for document-based question answering with **multi-document support**, **similarity-based refusal**, and **comprehensive testing**.

Built with a focus on **system design**, **correctness**, and **production thinking** — not just calling an LLM.

---

##  Key Features

###  Multi-Document Support
- Upload and query **100+ documents independently**
- Each document gets unique ID
- Queries are isolated per document (no cross-contamination)

###  Deterministic Refusal Logic
- **Similarity threshold (0.65)** prevents hallucinations
- System refuses to answer when confidence is low
- Honest "I don't know" instead of confident lies

###  Production-Grade Architecture
- **Layered design:** API → Workflow → Memory → LLM
- **Structured logging:** JSON logs with request tracing
- **Input validation:** Pydantic models, error handling
- **Performance tuning:** 500-word chunks, top-5 retrieval, ~1-2s latency

###  Comprehensive Testing
- **57 test cases** covering API, workflow, and edge cases
- Tests for concurrent uploads, special characters, boundary conditions
- Validates refusal logic and multi-document isolation

---

##  System Architecture
```
Client (UI/API)
    ↓
API Layer (FastAPI)
    ↓
Workflow Layer (RAG Pipeline + Refusal Logic)
    ↓
Memory Layer (Chunking, Embeddings, Vector Search) + LLM Layer (OpenAI)
    ↓
Observability Layer (Structured Logging, Tracing)
```

**See [`ARCHITECTURE.md`](ARCHITECTURE.md) for detailed explanation.**

---

##  Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- OpenAI API key

### 2. Set Environment Variable
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Start the API Server
```bash
uvicorn app.main:app --reload
```

API will start at: `http://127.0.0.1:8000`

### 4. Start the UI (Optional)
```bash
streamlit run app/ui/app.py
```

UI will open at: `http://localhost:8501`

---

##  Usage Examples

### Upload a Document
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@contract.pdf"
```

**Response:**
```json
{
  "document_id": "doc_abc123def456",
  "filename": "contract.pdf",
  "chunks_created": 47,
  "message": "Document uploaded and indexed successfully"
}
```

### Ask a Question
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_abc123def456",
    "question": "What is the contract effective date?"
  }'
```

**Response (High Confidence):**
```json
{
  "answer": "The contract effective date is January 15, 2024.",
  "document_id": "doc_abc123def456",
  "confidence_score": 0.87,
  "refused": false,
  "sources_used": 5,
  "reasoning": null
}
```

**Response (Low Confidence - Refused):**
```json
{
  "answer": "I don't have enough confident information in the document to answer this question. The most relevant content has only 42% confidence, which is below the 65% threshold.",
  "document_id": "doc_abc123def456",
  "confidence_score": 0.42,
  "refused": true,
  "sources_used": 5,
  "reasoning": "Top similarity score (0.420) below threshold (0.65)"
}
```

### List All Documents
```bash
curl "http://127.0.0.1:8000/documents"
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": "doc_abc123",
      "filename": "contract.pdf",
      "chunks_count": 47,
      "upload_timestamp": "2024-01-30T10:15:00Z"
    },
    {
      "document_id": "doc_xyz789",
      "filename": "invoice.pdf",
      "chunks_count": 12,
      "upload_timestamp": "2024-01-30T10:20:00Z"
    }
  ],
  "total_documents": 2,
  "total_chunks": 59
}
```

### Delete a Document
```bash
curl -X DELETE "http://127.0.0.1:8000/documents/doc_abc123"
```

### Health Check
```bash
curl "http://127.0.0.1:8000/health"
```

---

##  Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

**Expected output:**
```
tests/test_api.py::TestHealthEndpoint::test_health_check_no_documents PASSED
tests/test_api.py::TestUploadEndpoint::test_upload_valid_pdf PASSED
tests/test_workflow.py::TestSimilarityThresholdRefusal::test_refuse_when_no_chunks_retrieved PASSED
...
===================== 57 passed in 12.34s =====================
```

### Run Specific Test File
```bash
pytest tests/test_workflow.py -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=app --cov-report=html
```

---

## Configuration

All tunable parameters are in [`app/config.py`](app/config.py):
```python
# Document Processing
CHUNK_SIZE = 500           # Words per chunk
CHUNK_OVERLAP = 100        # Overlap between chunks

# Retrieval
TOP_K = 5                  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.65  # Refusal threshold

# LLM
LLM_MODEL = "gpt-4o-mini"  # OpenAI model
LLM_TEMPERATURE = 0.2      # Low temp for factual answers

# Limits
MAX_FILE_SIZE_MB = 10
```

**To tune behavior:** Edit `app/config.py` and restart the server.

---

##  Project Structure
```
.
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints
│   ├── llm/
│   │   └── client.py          # OpenAI client
│   ├── memory/
│   │   ├── loader.py          # PDF text extraction
│   │   ├── chunker.py         # Text chunking
│   │   ├── embedder.py        # Embedding generation
│   │   ├── store.py           # Vector store (FAISS)
│   │   └── retriever.py       # Similarity search
│   ├── workflow/
│   │   └── document_qa.py     # RAG pipeline + refusal logic
│   ├── observability/
│   │   └── logger.py          # Structured logging
│   ├── ui/
│   │   └── app.py             # Streamlit UI
│   ├── config.py              # Configuration
│   ├── models.py              # Pydantic models
│   └── main.py                # FastAPI app
├── tests/
│   ├── conftest.py            # Test fixtures
│   ├── test_api.py            # API tests (24 tests)
│   ├── test_workflow.py       # Workflow tests (20 tests)
│   └── test_edge_cases.py     # Edge cases (13 tests)
├── logs/
│   └── app.log                # JSON logs
├── ARCHITECTURE.md            # System design documentation
├── DESIGN_DECISIONS.md        # Trade-off analysis
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

---

##  How It Works

### 1. Upload Flow
```
PDF → Extract Text → Split into Chunks → Generate Embeddings → Store in FAISS
```

### 2. Query Flow
```
Question → Generate Embedding → Search FAISS → Check Similarity Threshold
    ↓ (if above 0.65)
Build Prompt → Call OpenAI → Return Answer
    ↓ (if below 0.65)
Return "I don't know" (REFUSE)
```

### 3. Refusal Logic (Critical Feature)
```python
if top_similarity_score < 0.65:
    return {
        "answer": "I don't have enough confident information...",
        "refused": True,
        "confidence_score": 0.42,
        "reasoning": "Below threshold"
    }
```

**This prevents hallucinations.**

---

##  What I Learned

### System Design
- **Layered architecture** matters more than prompt engineering
- **Refusal logic** builds user trust (better to say "I don't know" than lie)
- **Clean boundaries** make debugging 10x easier

### Production Thinking
- **Similarity thresholds** prevent hallucinations (empirically tuned to 0.65)
- **Structured logging** is essential (JSON logs saved my life debugging)
- **Testing** catches bugs before users do 

### Trade-offs
- **In-memory FAISS** → Fast & simple, but not persistent (acceptable for v1)
- **Single-instance** → Predictable, but doesn't scale horizontally (okay for 100 docs)
- **gpt-4o-mini** → Cheap & fast, but not best quality (95% accuracy is enough)

**See [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) for full analysis.**

---

##  Current Limitations (v1 Scope)

These are **intentional scope decisions**, not bugs:

-  **In-memory storage** - Data lost on restart (use persistent DB in production)
-  **No authentication** - Anyone can use API (add API keys in v2)
-  **Single instance** - No horizontal scaling (fine for 100 docs)
-  **No caching** - Same question re-embeds every time (add Redis in v2)

---

## Future Enhancements (v2+)

**If this were going to production:**

1. **Persistent Storage**
   - Replace FAISS → ChromaDB or Pinecone
   - Store documents in PostgreSQL

2. **Authentication & Security**
   - API key management
   - Rate limiting
   - User accounts

3. **Performance Optimizations**
   - Cache embeddings (Redis)
   - Batch processing
   - Async LLM calls

4. **Monitoring & Observability**
   - Prometheus metrics
   - Grafana dashboards
   - Alert on high refusal rate

5. **Advanced Features**
   - Multi-document search (query across all docs)
   - Conversation history
   - Citation with page numbers

**Architecture supports all of these without rewrite!**

---

##  Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, data flow, layer responsibilities
- **[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)** - Trade-offs, alternatives, tuning process

---
## Contributing

This is a portfolio project, but feedback is welcome!

**To report issues:**
1. Check if it's a known limitation (see above)
2. Open an issue with:
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant logs from `logs/app.log`

---

##  License

MIT License - See LICENSE file for details.

---

##  Acknowledgments

Built as a learning project to understand:
- How production RAG systems work
- How to structure AI applications
- How to make systems debuggable and testable

**Technologies used:**
- FastAPI (API framework)
- OpenAI GPT-4o-mini (LLM)
- FAISS (vector search)
- Sentence Transformers (embeddings)
- pytest (testing)
- Streamlit (UI)

---

##  Contact

**Gopal Khandare**  
AI Engineer (Generative AI & Backend Systems)  
[LinkedIn](https://linkedin.com/in/your-profile)  
[GitHub](https://github.com/gopal861)

