# app/config.py
"""
Configuration for AI Document Understanding System.

This file centralizes all tunable parameters for the RAG pipeline.
Changes here affect system behavior without code modifications.
"""

# ========== DOCUMENT PROCESSING ==========


import os
from dotenv import load_dotenv

load_dotenv()


# Chunk configuration (matches resume: "chunk sizes 500-800 tokens")
CHUNK_SIZE = 700  # words per chunk

CHUNK_OVERLAP = 120 # overlap between chunks to preserve context

# File upload limits
MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_EXTENSIONS = [".pdf"]


# ========== EMBEDDING CONFIGURATION ==========

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# ========== RETRIEVAL CONFIGURATION ==========

# Retrieval depth 
TOP_K = 8 # Number of chunks to retrieve

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



# ============================================================
# INGESTION SAFETY LIMITS
# ============================================================

# Maximum characters allowed per document
MAX_DOCUMENT_CHARACTERS = 2_000_000
# Maximum HTML pages to crawl
MAX_HTML_PAGES = 20


# ============================================================
# QDRANT CONFIGURATION
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")


# Allowed domains for URL ingestion

ALLOWED_DOMAINS = [

    "github.com",
    "raw.githubusercontent.com",

    "platform.openai.com",
    "ai.google.dev",

    "docs.python.org",
    "fastapi.tiangolo.com",
    "huggingface.co" ,
    "docs.anthropic.com"

]
