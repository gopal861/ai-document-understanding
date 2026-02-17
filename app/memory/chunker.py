# app/memory/chunker.py

import logging
from typing import List

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_DOCUMENT_CHARACTERS,
)

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Production-grade bounded chunker.

    Architecture contract preserved:
    loader → chunker → embedder → vector_store

    Guarantees:
    • deterministic chunk generation
    • bounded memory usage
    • no infinite loops
    • no empty chunks
    • production observability
    """

    # ============================================================
    # SAFETY CHECKS
    # ============================================================

    if not text:
        logger.warning("Chunking skipped: empty text")
        return []

    text = text.strip()

    if not text:
        logger.warning("Chunking skipped: whitespace text")
        return []

    # enforce global character limit safety
    if len(text) > MAX_DOCUMENT_CHARACTERS:
        logger.warning(
            "Text exceeds max character limit, truncating",
            extra={
                "original_length": len(text),
                "max_allowed": MAX_DOCUMENT_CHARACTERS,
            },
        )
        text = text[:MAX_DOCUMENT_CHARACTERS]

    if size <= 0:
        raise ValueError(f"Invalid chunk size: {size}")

    if overlap < 0:
        raise ValueError(f"Invalid chunk overlap: {overlap}")

    if overlap >= size:
        raise ValueError(
            f"Overlap must be smaller than chunk size "
            f"(overlap={overlap}, size={size})"
        )

    # ============================================================
    # TOKENIZATION
    # ============================================================

    words = text.split()

    if not words:
        logger.warning("Chunking skipped: no words found")
        return []

    total_words = len(words)

    chunks = []

    start = 0

    step = size - overlap

    # ============================================================
    # CHUNK GENERATION LOOP
    # ============================================================

    while start < total_words:

        end = start + size

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunk = " ".join(chunk_words).strip()

        if chunk:
            chunks.append(chunk)

        # move forward safely
        start += step

        # prevent infinite loop
        if step <= 0:
            break

    # ============================================================
    # OBSERVABILITY
    # ============================================================

    logger.info(
        "Chunking completed",
        extra={
            "total_words": total_words,
            "chunk_size": size,
            "overlap": overlap,
            "chunks_created": len(chunks),
        },
    )

    return chunks
