import faiss
import numpy as np
import logging
import os
import json
import uuid

from typing import List, Dict, Optional

from qdrant_client.http.models import PointStruct

from app.config import (
    MAX_CHUNKS_PER_DOCUMENT,
    MAX_DOCUMENTS_IN_MEMORY,
    TOP_K,
    QDRANT_COLLECTION,
)

from app.memory.qdrant_client import QdrantVectorDB


logger = logging.getLogger(__name__)


class VectorStore:

    _INDEX_PATH = "storage/faiss.index"
    _METADATA_PATH = "storage/metadata.json"

    # ============================================================
    # INIT
    # ============================================================

    def __init__(self, dim: int):

        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        self._dim = dim

        self._chunks: List[Dict] = []

        self._doc_chunk_count: Dict[str, int] = {}

        self._index = None

        # Initialize Qdrant backend
        self._qdrant = QdrantVectorDB(dim)

        # Load FAISS fallback
        self._load_from_disk()

        if self._index is None:

            self._index = faiss.IndexFlatIP(dim)

            logger.info(
                "New FAISS fallback index created",
                extra={"dimension": dim},
            )

        logger.info(
            "VectorStore initialized",
            extra={
                "dimension": dim,
                "faiss_vectors": self._index.ntotal,
            },
        )

    # ============================================================
    # LOAD FROM DISK
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

                self._doc_chunk_count = data.get(
                    "doc_chunk_count",
                    {},
                )

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
    # SAVE TO DISK
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
                "FAISS fallback persisted",
                extra={"vectors": self._index.ntotal},
            )

        except Exception as e:

            logger.error(
                "Persistence failed",
                extra={"error": str(e)},
            )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _ensure_numpy(self, embeddings) -> np.ndarray:

        if embeddings is None:
            raise ValueError("Embeddings cannot be None")

        if isinstance(embeddings, list):

            embeddings = np.array(
                embeddings,
                dtype="float32",
            )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")

        if embeddings.shape[1] != self._dim:
            raise ValueError("Embedding dimension mismatch")

        return embeddings

    def _normalize(self, vectors: np.ndarray):

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        return vectors / np.clip(norms, 1e-10, None)

    # ============================================================
    # ADD DOCUMENT
    # ============================================================

    def add(
        self,
        embeddings,
        chunks: List[str],
        doc_id: str,
    ):

        embeddings = self._ensure_numpy(embeddings)

        embeddings = self._normalize(
            embeddings.astype("float32")
        )

        existing_chunks = self._doc_chunk_count.get(doc_id, 0)

        start_idx = len(self._chunks)

        points = []

        for i, embedding in enumerate(embeddings):

            global_idx = start_idx + i

            points.append(

                PointStruct(

                    id=str(uuid.uuid4()),

                    vector=embedding.tolist(),

                    payload={

                        "text": chunks[i],

                        "doc_id": doc_id,

                        "chunk_idx": existing_chunks + i,

                        "global_idx": global_idx,

                    },

                )

            )

        # BULK UPSERT (FAST)
        self._qdrant._client.upsert(

            collection_name=QDRANT_COLLECTION,

            points=points,

        )

        # FAISS fallback
        self._index.add(embeddings)

        for i, chunk in enumerate(chunks):

            self._chunks.append({

                "text": chunk,

                "doc_id": doc_id,

                "chunk_idx": existing_chunks + i,

                "global_idx": start_idx + i,

            })

        self._doc_chunk_count[doc_id] = (

            existing_chunks + len(chunks)

        )

        self._save_to_disk()

        logger.info(
            "Document indexed successfully",
            extra={
                "doc_id": doc_id,
                "vectors": len(chunks),
            },
        )

    # ============================================================
    # QUERY (FIXED FOR QDRANT CLIENT 1.16.2)
    # ============================================================

    def query(
        self,
        embedding,
        top_k=TOP_K,
        doc_id=None,
        deleted_docs=None,
    ):

        embedding = self._ensure_numpy(embedding)

        embedding = self._normalize(
            embedding.astype("float32")
        )

        # CORRECT METHOD FOR 1.16.2
        hits = self._qdrant._client.search(

            collection_name=QDRANT_COLLECTION,

            query_vector=embedding[0].tolist(),

            limit=top_k,

        )

        results = []

        for hit in hits:

            payload = hit.payload or {}

            if deleted_docs and payload.get("doc_id") in deleted_docs:
                continue

            if doc_id and payload.get("doc_id") != doc_id:
                continue

            results.append({

                "text": payload.get("text"),

                "doc_id": payload.get("doc_id"),

                "chunk_idx": payload.get("chunk_idx"),

                "similarity_score": float(hit.score),

            })

        return results

    # ============================================================
    # STATS
    # ============================================================

    def get_stats(self):

        return {

            "total_chunks": len(self._chunks),

            "total_vectors": self._index.ntotal,

            "documents": dict(self._doc_chunk_count),

        }

    def document_exists(self, doc_id):

        return doc_id in self._doc_chunk_count

    def document_count(self):

        return len(self._doc_chunk_count)

