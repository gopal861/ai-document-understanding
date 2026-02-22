import logging

from qdrant_client import QdrantClient

from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,   # ✅ REQUIRED FIX
)

from app.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
)

logger = logging.getLogger(__name__)


class QdrantVectorDB:
    """
    Safe Qdrant client wrapper.

    Does NOT modify architecture contract.
    Only provides storage backend.
    """

    def __init__(self, dim: int):

        self._dim = dim

        self._client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60.0,
        )

        self._collection = QDRANT_COLLECTION

        self._ensure_collection()

        logger.info(
            "Qdrant client initialized",
            extra={
                "collection": self._collection,
                "dimension": dim,
            },
        )


    def _ensure_collection(self):
        """
        Ensures collection exists AND required payload indexes exist.

        This preserves architecture contract and only adds index safety.
        """

        collections = self._client.get_collections().collections

        exists = any(
            c.name == self._collection
            for c in collections
        )

        # ============================================================
        # Create collection if missing
        # ============================================================

        if not exists:

            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dim,
                    distance=Distance.COSINE,
                ),
            )

            logger.info(
                "Qdrant collection created",
                extra={"collection": self._collection},
            )

        # ============================================================
        # CRITICAL FIX: Ensure doc_id payload index exists
        # Required for delete, filter, and retrieval correctness
        # ============================================================

        try:

            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="doc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            logger.info(
                "Payload index ensured for doc_id",
                extra={"collection": self._collection},
            )

        except Exception as e:
            # Safe to ignore if index already exists
            logger.debug(
                "Payload index already exists or skipped",
                extra={"error": str(e)},
            )


    def health_check(self):

        return self._client.get_collections()
