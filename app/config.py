# app/config.py
"""
Configuration for AI Document Understanding System.

This file centralizes all tunable parameters for the RAG pipeline.
Changes here affect system behavior without code modifications.
"""

# ========== DOCUMENT PROCESSING ==========

# Chunk configuration (matches resume: "chunk sizes 500-800 tokens")
CHUNK_SIZE = 700  # words per chunk

CHUNK_OVERLAP = 120 # overlap between chunks to preserve context

# File upload limits
MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_EXTENSIONS = [".pdf"]


# ========== EMBEDDING CONFIGURATION ==========

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


# ========== RETRIEVAL CONFIGURATION ==========

# Retrieval depth 
TOP_K = 15 # Number of chunks to retrieve

# Similarity threshold for refusal 
SIMILARITY_THRESHOLD = 0.45


# - Below this threshold → refuse to answer
# - Above this threshold → generate answer
# - Trade-off: Higher = fewer false positives, more refusals
#              Lower = more answers, risk of hallucination


# ========== LLM CONFIGURATION ==========

# Model selection
LLM_MODEL = "gpt-4o-mini"  # Fast, cheap, good for most queries
# Alternative: "gpt-4o" (better quality, slower, more expensive)

# Generation parameters
LLM_TEMPERATURE = 0.2  # Low temperature for factual answers
LLM_MAX_TOKENS = 500  # Limit response length


# ========== SYSTEM CONSTRAINTS ==========

# Latency target 
TARGET_LATENCY_SECONDS = 2.0




# Memory limits
MAX_DOCUMENTS_IN_MEMORY = 100  # Safety limit for single-instance deployment
MAX_CHUNKS_PER_DOCUMENT = 500 # Prevent memory exhaustion from huge documents


# ========== DESIGN TRADE-OFFS (DOCUMENTED) ==========

"""
TRADE-OFF DECISIONS:



1. In-memory FAISS (not persistent DB):
   - Trade-off: Fast retrieval, simple deployment
   - Limitation: Data lost on restart, limited to single instance
   

2. Single-instance deployment:
   - Trade-off: Predictable behavior, easier debugging
   - Limitation: No horizontal scaling
   
"""

# ============================================================
# INGESTION SAFETY LIMITS
# ============================================================

# Maximum characters allowed per document
MAX_DOCUMENT_CHARACTERS = 2_000_000
# Maximum HTML pages to crawl
MAX_HTML_PAGES = 20

# Allowed domains for URL ingestion
ALLOWED_DOMAINS = {
    "platform.openai.com",
    "ai.google.dev",
    "docs.anthropic.com",
    "huggingface.co",
    "github.com",
}
