# app/memory/store.py

import faiss
import numpy as np
import logging
import os
import json
from typing import List, Dict, Optional

from app.config import (
    MAX_CHUNKS_PER_DOCUMENT,
    MAX_DOCUMENTS_IN_MEMORY,
    TOP_K,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Production-grade FAISS cosine similarity vector store.

    Architecture contract preserved:
    embedder → vector_store → retriever → LLM

    Guarantees:
    • bounded memory growth
    • safe ingestion limits
    • stable similarity ranking
    • production observability
    • retrieval correctness
    """

    # ============================================================
    # PERSISTENCE PATHS (NEW)
    # ============================================================

    _INDEX_PATH = "storage/faiss.index"
    _METADATA_PATH = "storage/metadata.json"

    # ============================================================
    # INIT
    # ============================================================

    def __init__(self, dim: int):

        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        self._dim = dim

        # chunk metadata
        self._chunks: List[Dict] = []

        # document → chunk count
        self._doc_chunk_count: Dict[str, int] = {}

        # index will be loaded from disk if exists
        self._index = None

        # attempt load
        self._load_from_disk()

        # create new index if none exists
        if self._index is None:

            self._index = faiss.IndexFlatIP(dim)

            logger.info(
                "New FAISS index created",
                extra={"dimension": dim},
            )

        logger.info(
            "VectorStore initialized",
            extra={
                "dimension": dim,
                "vectors_loaded": self._index.ntotal,
            },
        )

    # ============================================================
    # PERSISTENCE LOAD (NEW)
    # ============================================================

    def _load_from_disk(self):

        os.makedirs("storage", exist_ok=True)

        if os.path.exists(self._INDEX_PATH):

            try:

                self._index = faiss.read_index(self._INDEX_PATH)

                logger.info(
                    "FAISS index loaded",
                    extra={"vectors": self._index.ntotal},
                )

            except Exception as e:

                logger.error(
                    "FAISS load failed",
                    extra={"error": str(e)},
                )

                self._index = None

        if os.path.exists(self._METADATA_PATH):

            try:

                with open(self._METADATA_PATH, "r") as f:

                    data = json.load(f)

                self._chunks = data.get("chunks", [])

                self._doc_chunk_count = data.get("doc_chunk_count", {})

                logger.info(
                    "Metadata loaded",
                    extra={"chunks": len(self._chunks)},
                )

            except Exception as e:

                logger.error(
                    "Metadata load failed",
                    extra={"error": str(e)},
                )

    # ============================================================
    # PERSISTENCE SAVE (NEW)
    # ============================================================

    def _save_to_disk(self):

        try:

            os.makedirs("storage", exist_ok=True)

            faiss.write_index(self._index, self._INDEX_PATH)

            with open(self._METADATA_PATH, "w") as f:

                json.dump(
                    {
                        "chunks": self._chunks,
                        "doc_chunk_count": self._doc_chunk_count,
                    },
                    f,
                )

            logger.info(
                "VectorStore persisted",
                extra={"vectors": self._index.ntotal},
            )

        except Exception as e:

            logger.error(
                "Persistence failed",
                extra={"error": str(e)},
            )

    # ============================================================
    # INTERNAL SAFETY
    # ============================================================

    def _ensure_numpy(self, embeddings) -> np.ndarray:

        if embeddings is None:
            raise ValueError("Embeddings cannot be None")

        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype="float32")

        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be numpy array or list")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")

        if embeddings.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {self._dim}, got {embeddings.shape[1]}"
            )

        return embeddings

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        return vectors / np.clip(norms, 1e-10, None)

    # ============================================================
    # ADD DOCUMENT
    # ============================================================

    def add(
        self,
        embeddings,
        chunks: List[str],
        doc_id: str,
    ) -> None:

        if not doc_id:
            raise ValueError("doc_id required")

        if not chunks:
            raise ValueError("No chunks provided")

        embeddings = self._ensure_numpy(embeddings)

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "Chunks count and embeddings count mismatch"
            )

        if doc_id not in self._doc_chunk_count:

            if len(self._doc_chunk_count) >= MAX_DOCUMENTS_IN_MEMORY:

                raise ValueError(
                    f"Max documents limit reached "
                    f"({MAX_DOCUMENTS_IN_MEMORY})"
                )

        existing_chunks = self._doc_chunk_count.get(doc_id, 0)

        incoming_chunks = len(chunks)

        if existing_chunks + incoming_chunks > MAX_CHUNKS_PER_DOCUMENT:

            raise ValueError(
                f"Max chunks per document exceeded "
                f"({MAX_CHUNKS_PER_DOCUMENT})"
            )

        embeddings = embeddings.astype("float32")

        embeddings = self._normalize(embeddings)

        start_idx = len(self._chunks)

        self._index.add(embeddings)

        for i, chunk in enumerate(chunks):

            self._chunks.append({
                "text": chunk,
                "doc_id": doc_id,
                "chunk_idx": existing_chunks + i,
                "global_idx": start_idx + i,
            })

        self._doc_chunk_count[doc_id] = (
            existing_chunks + incoming_chunks
        )

        logger.info(
            "Document indexed",
            extra={
                "doc_id": doc_id,
                "chunks_added": incoming_chunks,
                "total_chunks": self._doc_chunk_count[doc_id],
                "total_vectors": self._index.ntotal,
            },
        )

        # SAVE PERSISTENCE (NEW)
        self._save_to_disk()

    # ============================================================
    # QUERY
    # ============================================================

    def query(
        self,
        embedding,
        top_k: int = TOP_K,
        doc_id: Optional[str] = None,
        deleted_docs: Optional[set] = None,
    ) -> List[Dict]:

        if self._index.ntotal == 0:
            logger.warning("Query on empty vector store")
            return []

        embedding = self._ensure_numpy(embedding)

        embedding = embedding.astype("float32")

        embedding = self._normalize(embedding)

        scores, indices = self._index.search(
            embedding,
            min(top_k * 5, self._index.ntotal),
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0 or idx >= len(self._chunks):
                continue

            chunk = self._chunks[idx]

            if deleted_docs and chunk["doc_id"] in deleted_docs:
                continue

            if doc_id and chunk["doc_id"] != doc_id:
                continue

            results.append({
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "chunk_idx": chunk["chunk_idx"],
                "similarity_score": float(score),
            })

            if len(results) >= top_k:
                break

        logger.info(
            "Vector query executed",
            extra={
                "doc_id": doc_id,
                "results_returned": len(results),
                "top_score": results[0]["similarity_score"]
                if results else None,
            },
        )

        return results

    # ============================================================
    # OBSERVABILITY
    # ============================================================

    def get_stats(self) -> Dict:

        return {
            "total_chunks": len(self._chunks),
            "total_vectors": self._index.ntotal,
            "documents": dict(self._doc_chunk_count),
        }

    def document_exists(self, doc_id: str) -> bool:

        return doc_id in self._doc_chunk_count

    def document_count(self) -> int:

        return len(self._doc_chunk_count)



