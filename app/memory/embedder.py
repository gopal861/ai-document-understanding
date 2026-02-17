# app/memory/embedder.py

"""
Production-grade embedding wrapper with batching optimization.

Architecture contract:
chunker → embedder → vector_store

Guarantees:
• Deterministic embeddings
• Always returns numpy float32 array
• Always normalized (cosine-ready)
• Batched processing for performance
• Fully observable via logs
• Safe bounded memory usage
"""

import logging
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
DEFAULT_EMBED_BATCH_SIZE = 32


from app.config import (
    EMBEDDING_MODEL,
    MAX_CHUNKS_PER_DOCUMENT,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION (SAFE DEFAULT)
# ============================================================

# Optimal batch size for CPU systems
# Safe for laptops and small servers



class Embedder:
    """
    Production-safe embedding generator.

    Responsibilities:
    • Load embedding model
    • Generate normalized embeddings
    • Batch processing for speed
    • Enforce system limits
    • Guarantee FAISS-compatible output
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        logger.info(
            "Initializing embedding model",
            extra={"model": EMBEDDING_MODEL}
        )

        try:

            self._model = SentenceTransformer(EMBEDDING_MODEL)

            self._dimension = (
                self._model.get_sentence_embedding_dimension()
            )

            logger.info(
                "Embedding model loaded",
                extra={
                    "model": EMBEDDING_MODEL,
                    "dimension": self._dimension,
                }
            )

        except Exception as e:

            logger.critical(
                "Embedding model initialization failed",
                extra={"error": str(e)}
            )

            raise RuntimeError(
                f"Failed to load embedding model: {e}"
            )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def embed(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    ) -> np.ndarray:
        """
        Generate embeddings with optimized batching.

        Guarantees:
        • Always numpy.ndarray
        • Always float32
        • Always normalized
        • Always correct shape
        """

        if not texts:

            logger.warning("Empty embedding request")

            return np.empty(
                (0, self._dimension),
                dtype="float32"
            )

        if len(texts) > MAX_CHUNKS_PER_DOCUMENT:

            raise ValueError(
                f"Chunk count exceeds MAX_CHUNKS_PER_DOCUMENT "
                f"({MAX_CHUNKS_PER_DOCUMENT})"
            )

        total = len(texts)

        logger.info(
            "Embedding started",
            extra={
                "chunks": total,
                "batch_size": batch_size,
            }
        )

        try:

            all_embeddings = []

            # ====================================================
            # BATCH PROCESSING LOOP
            # ====================================================

            for start in range(0, total, batch_size):

                end = min(start + batch_size, total)

                batch = texts[start:end]

                batch_embeddings = self._model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                batch_embeddings = batch_embeddings.astype("float32")

                all_embeddings.append(batch_embeddings)

            # ====================================================
            # CONCATENATE ALL BATCHES
            # ====================================================

            embeddings = np.vstack(all_embeddings)

            logger.info(
                "Embedding completed",
                extra={
                    "chunks": total,
                    "dimension": self._dimension,
                    "shape": embeddings.shape,
                }
            )

            return embeddings

        except Exception as e:

            logger.error(
                "Embedding generation failed",
                extra={"error": str(e)}
            )

            raise RuntimeError(
                f"Embedding generation failed: {e}"
            )

    # ============================================================
    # ACCESSORS
    # ============================================================

    def get_dimension(self) -> int:
        """
        Required by VectorStore initialization.
        """
        return self._dimension

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self) -> dict:
        """
        Used for system observability.
        """

        return {
            "model": EMBEDDING_MODEL,
            "dimension": self._dimension,
            "status": "healthy"
        }
