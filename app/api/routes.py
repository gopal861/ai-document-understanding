from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Request
import uuid
import logging
import time
import threading
import os
import json

from datetime import datetime
from pathlib import Path
from typing import Dict

from app.observability.metrics import metrics_tracker
from app.observability.posthog_client import posthog_client

from app.models import (
    AskRequest,
    AskResponse,
    UploadResponse,
    ListDocumentsResponse,
    DeleteDocumentResponse,
    DocumentInfo,
    HealthResponse,
)

from app.memory.loader import load_text
from app.memory.chunker import chunk_text
from app.memory.embedder import Embedder
from app.memory.store import VectorStore
from app.memory.retriever import retrieve
from app.workflow.document_qa import answer_question
from app.llm.multi_model_client import MultiModelLLMClient


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# GLOBAL SINGLETONS (ARCHITECTURE CONTRACT PRESERVED)
# ============================================================

embedder = Embedder()

vector_store = VectorStore(
    dim=embedder.get_dimension()
)

llm_client = MultiModelLLMClient()


# ============================================================
# DOCUMENT REGISTRY (PERSISTENT)
# ============================================================

DOCUMENT_REGISTRY_PATH = "storage/document_registry.json"

document_registry: Dict[str, dict] = {}


def load_document_registry():

    global document_registry

    if not os.path.exists(DOCUMENT_REGISTRY_PATH):
        logger.info("Document registry file not found. Starting fresh.")
        return

    try:

        with open(DOCUMENT_REGISTRY_PATH, "r") as f:
            data = json.load(f)

        restored = {}

        for doc_id, meta in data.items():

            timestamp = meta.get("upload_timestamp")

            if timestamp:
                try:
                    meta["upload_timestamp"] = datetime.fromisoformat(timestamp)
                except Exception:
                    meta["upload_timestamp"] = None

            restored[doc_id] = meta

        document_registry.clear()
        document_registry.update(restored)

        logger.info(
            "Document registry loaded",
            extra={"documents": len(document_registry)}
        )

    except Exception as e:

        logger.error(
            "Document registry load failed",
            extra={"error": str(e)}
        )


def save_document_registry():

    try:

        os.makedirs("storage", exist_ok=True)

        serializable = {}

        for doc_id, meta in document_registry.items():

            timestamp = meta.get("upload_timestamp")

            serializable[doc_id] = {
                "filename": meta.get("filename"),
                "chunks_count": meta.get("chunks_count"),
                "upload_timestamp":
                    timestamp.isoformat() if isinstance(timestamp, datetime) else None,
            }

        with open(DOCUMENT_REGISTRY_PATH, "w") as f:
            json.dump(serializable, f)

        logger.info("Document registry saved")

    except Exception as e:

        logger.error(
            "Document registry save failed",
            extra={"error": str(e)}
        )


# Load registry on startup
load_document_registry()


# ============================================================
# INGESTION LOCK (CRITICAL SAFETY)
# ============================================================

ingestion_lock = threading.Lock()


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FILE_SIZE_MB = 10

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

INGESTION_MAX_RETRIES = 3
INGESTION_RETRY_DELAY = 2


# ============================================================
# HELPERS
# ============================================================

def generate_document_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"


def validate_file_size(content: bytes):

    size_mb = len(content) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.2f}MB",
        )


def load_text_with_retry(source: str) -> str:

    last_error = None

    for attempt in range(INGESTION_MAX_RETRIES):

        try:

            logger.info(
                "Loading document",
                extra={
                    "source": source,
                    "attempt": attempt + 1,
                },
            )

            text = load_text(source)

            if text:
                return text

        except Exception as e:

            last_error = e

            logger.warning(
                "Document load failed",
                extra={
                    "source": source,
                    "attempt": attempt + 1,
                    "error": str(e),
                },
            )

        time.sleep(INGESTION_RETRY_DELAY)

    raise last_error


# ============================================================
# HEALTH
# ============================================================

@router.get("/health", response_model=HealthResponse)
def health_check():

    stats = vector_store.get_stats()

    return HealthResponse(
        status="healthy",
        total_documents=len(stats["documents"]),
        total_chunks=stats["total_chunks"],
        total_vectors=stats["total_vectors"],
    )


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(None),
):

    if not file and not url:

        raise HTTPException(
            status_code=400,
            detail="Provide file or URL",
        )

    document_id = generate_document_id()

    start_time = time.time()

    try:

        if url:

            filename = url
            text = load_text_with_retry(url)

        else:

            file_bytes = await file.read()

            validate_file_size(file_bytes)

            file_path = UPLOAD_DIR / f"{document_id}.pdf"

            with file_path.open("wb") as buffer:
                buffer.write(file_bytes)

            filename = file.filename

            text = load_text_with_retry(str(file_path))

        if not text:
            raise HTTPException(status_code=400, detail="No text extracted")

        chunks = chunk_text(text)

        embeddings = embedder.embed(chunks)

        with ingestion_lock:

            vector_store.add(
                embeddings=embeddings,
                chunks=chunks,
                doc_id=document_id,
            )

        document_registry[document_id] = {
            "filename": filename,
            "chunks_count": len(chunks),
            "upload_timestamp": datetime.utcnow(),
        }

        save_document_registry()

        latency = time.time() - start_time

        logger.info(
            "Document ingestion complete",
            extra={"doc_id": document_id}
        )

        posthog_client.track_document_upload(
            distinct_id=request.state.request_id,
            document_id=document_id,
            filename=filename,
            chunks=len(chunks),
            latency=latency,
        )

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            chunks_created=len(chunks),
        )

    except Exception as e:

        posthog_client.track_error(
            distinct_id=request.state.request_id,
            error_type=type(e).__name__,
            error_message=str(e),
            endpoint="/upload",
        )

        raise


# ============================================================
# ASK QUESTION
# ============================================================

@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request):

    start_time = time.time()

    try:

        if payload.document_id not in document_registry:

            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )

        def retrieve_fn(question: str, top_k: int):

            results = retrieve(
                question=question,
                embedder=embedder,
                store=vector_store,
                top_k=top_k,
                doc_id=payload.document_id,
            )

            posthog_client.track_retrieval(
                distinct_id=request.state.request_id,
                document_id=payload.document_id,
                chunks_retrieved=len(results),
                top_score=results[0]["similarity_score"] if results else None,
            )

            return results

        result = answer_question(
            question=payload.question,
            session_id=payload.document_id,
            retrieve_fn=retrieve_fn,
            llm_client=llm_client,
        )

        latency = time.time() - start_time

        posthog_client.track_question(
            distinct_id=request.state.request_id,
            document_id=payload.document_id,
            question=payload.question,
            latency=latency,
            success=True,
        )

        return AskResponse(**result)

    except Exception as e:

        posthog_client.track_error(
            distinct_id=request.state.request_id,
            error_type=type(e).__name__,
            error_message=str(e),
            endpoint="/ask",
        )

        raise


# ============================================================
# LIST DOCUMENTS
# ============================================================

@router.get("/documents", response_model=ListDocumentsResponse)
def list_documents():

    documents = []

    for doc_id, meta in document_registry.items():

        timestamp = meta.get("upload_timestamp")

        documents.append(
            DocumentInfo(
                document_id=doc_id,
                filename=meta.get("filename"),
                chunks_count=meta.get("chunks_count"),
                upload_timestamp=str(timestamp) if timestamp else None,
            )
        )

    return ListDocumentsResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(d.chunks_count for d in documents),
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================

@router.delete("/documents/{document_id}",
               response_model=DeleteDocumentResponse)
def delete_document(document_id: str):

    if document_id not in document_registry:

        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    del document_registry[document_id]

    save_document_registry()

    return DeleteDocumentResponse(
        document_id=document_id,
        message="Deleted",
        success=True,
    )


# ============================================================
# METRICS ENDPOINT
# ============================================================

@router.get("/metrics")
def get_metrics():

    return metrics_tracker.get_metrics()

