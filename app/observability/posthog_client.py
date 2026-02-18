# app/observability/posthog_client.py

"""
PostHog Observability Client

Architecture contract:
- Does NOT break existing logging
- Adds production-grade event tracking
- Uses request_id as fallback user_id
- Supports future frontend user_id automatically
- Never blocks API execution
"""

import os
import logging
from typing import Optional, Dict, Any

from posthog import Posthog


logger = logging.getLogger(__name__)


class PostHogClient:
    """
    Safe PostHog wrapper for production systems.

    Guarantees:
    - Never crashes API
    - Non-blocking tracking
    - Compatible with request_id tracing
    - Supports future user_id seamlessly
    """

    def __init__(self):

        self._enabled = False
        self._client: Optional[Posthog] = None

        api_key = os.getenv("POSTHOG_API_KEY")
        host = os.getenv("POSTHOG_HOST", "https://app.posthog.com")

        if not api_key:
            logger.warning(
                "PostHog disabled: POSTHOG_API_KEY not set"
            )
            return

        try:

            self._client = Posthog(
                project_api_key=api_key,
                host=host,
                timeout=5,
                flush_interval=1,
            )

            self._enabled = True

            logger.info(
                "PostHog client initialized",
                extra={"host": host}
            )

        except Exception as e:

            logger.error(
                "PostHog initialization failed",
                extra={"error": str(e)}
            )

            self._enabled = False


    # ==========================================================
    # INTERNAL SAFE TRACK
    # ==========================================================

    def _track(
        self,
        distinct_id: str,
        event: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Safe internal tracking method.
        Never throws exceptions.
        """

        if not self._enabled or not self._client:
            return

        try:

            self._client.capture(
                distinct_id=distinct_id,
                event=event,
                properties=properties or {},
            )

        except Exception as e:

            logger.warning(
                "PostHog tracking failed",
                extra={
                    "event": event,
                    "error": str(e),
                }
            )


    # ==========================================================
    # USER IDENTIFICATION
    # ==========================================================

    def identify_user(
        self,
        distinct_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Identify a user or request.

        distinct_id can be:
        - request_id (current system)
        - user_id (future frontend)
        """

        if not self._enabled or not self._client:
            return

        try:

            self._client.identify(
                distinct_id=distinct_id,
                properties=properties or {},
            )

        except Exception as e:

            logger.warning(
                "PostHog identify failed",
                extra={"error": str(e)}
            )


    # ==========================================================
    # DOCUMENT UPLOAD TRACKING
    # ==========================================================

    def track_document_upload(
        self,
        distinct_id: str,
        document_id: str,
        filename: str,
        chunks: int,
        latency: float,
    ):

        self._track(
            distinct_id,
            "document_uploaded",
            {
                "document_id": document_id,
                "filename": filename,
                "chunks": chunks,
                "latency_seconds": latency,
            },
        )


    # ==========================================================
    # QUESTION ASK TRACKING
    # ==========================================================

    def track_question(
        self,
        distinct_id: str,
        document_id: str,
        question: str,
        latency: float,
        success: bool,
    ):

        self._track(
            distinct_id,
            "question_asked",
            {
                "document_id": document_id,
                "question_length": len(question),
                "latency_seconds": latency,
                "success": success,
            },
        )


    # ==========================================================
    # RETRIEVAL TRACKING
    # ==========================================================

    def track_retrieval(
        self,
        distinct_id: str,
        document_id: str,
        chunks_retrieved: int,
        top_score: Optional[float],
    ):

        self._track(
            distinct_id,
            "retrieval_completed",
            {
                "document_id": document_id,
                "chunks_retrieved": chunks_retrieved,
                "top_score": top_score,
            },
        )


    # ==========================================================
    # ERROR TRACKING
    # ==========================================================

    def track_error(
        self,
        distinct_id: str,
        error_type: str,
        error_message: str,
        endpoint: str,
    ):

        self._track(
            distinct_id,
            "system_error",
            {
                "error_type": error_type,
                "error_message": error_message,
                "endpoint": endpoint,
            },
        )


# ==============================================================
# GLOBAL SINGLETON (IMPORTANT)
# ==============================================================

posthog_client = PostHogClient()
