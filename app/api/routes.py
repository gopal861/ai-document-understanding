from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Request
import uuid
import logging
import time
import threading
import os
import json
import asyncio

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
# GLOBAL SINGLETONS
# ============================================================

embedder = Embedder()

vector_store = VectorStore(
    dim=embedder.get_dimension()
)

llm_client = MultiModelLLMClient()


# ============================================================
# DOCUMENT REGISTRY
# ============================================================

DOCUMENT_REGISTRY_PATH = "storage/document_registry.json"

document_registry: Dict[str, dict] = {}


# ============================================================
# LOAD REGISTRY FROM DISK
# ============================================================

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
                    timestamp = datetime.fromisoformat(timestamp)
                except Exception:
                    timestamp = None

            restored[doc_id] = {
                "filename": meta.get("filename", doc_id),
                "chunks_count": meta.get("chunks_count", 0),
                "upload_timestamp": timestamp,
            }

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


# ============================================================
# CRITICAL FIX: REBUILD REGISTRY FROM VECTOR STORE
# ============================================================

def rebuild_registry_from_vector_store():

    global document_registry

    stats = vector_store.get_stats()

    documents = stats.get("documents", {})

    rebuilt_count = 0

    for doc_id, chunk_count in documents.items():

        if doc_id not in document_registry:

            document_registry[doc_id] = {
                "filename": doc_id,
                "chunks_count": chunk_count,
                "upload_timestamp": None,
            }

            rebuilt_count += 1

    if rebuilt_count > 0:

        save_document_registry()

        logger.info(
            "Registry rebuilt from vector store",
            extra={
                "rebuilt_documents": rebuilt_count,
                "total_documents": len(document_registry),
            },
        )


# ============================================================
# SAVE REGISTRY
# ============================================================

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


# ============================================================
# INITIALIZE REGISTRY CORRECTLY
# ============================================================

load_document_registry()

# CRITICAL FIX — rebuild from vector store
rebuild_registry_from_vector_store()


# ============================================================
# LOCK
# ============================================================

ingestion_lock = threading.Lock()


# ============================================================
# CONFIG
# ============================================================

MAX_FILE_SIZE_MB = 10

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

INGESTION_MAX_RETRIES = 3
INGESTION_RETRY_DELAY = 2


# ============================================================
# EXECUTOR SAFE LOADER
# ============================================================

async def load_text_with_retry(source: str) -> str:

    loop = asyncio.get_running_loop()

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

            text = await loop.run_in_executor(
                None,
                load_text,
                source
            )

            if text and text.strip():
                return text

            last_error = Exception("Empty text extracted")

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

        await asyncio.sleep(INGESTION_RETRY_DELAY)

    raise Exception(
        f"Document load failed after retries: {last_error}"
    )


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
# UPLOAD
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
     
    if file:
        content_length = request.headers.get("content-length")

        if content_length and int(content_length) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max allowed: {MAX_FILE_SIZE_MB}MB"
            )

    document_id = f"doc_{uuid.uuid4().hex[:12]}"

    start_time = time.time()

    try:

        if url:

            filename = url
            text = await load_text_with_retry(url)

        else:

            file_bytes = await file.read()

            file_path = UPLOAD_DIR / f"{document_id}.pdf"

            with file_path.open("wb") as buffer:
                buffer.write(file_bytes)

            filename = file.filename

            text = await load_text_with_retry(str(file_path))

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
# ASK
# ============================================================

@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request):

    if payload.document_id not in document_registry:

        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    def retrieve_fn(question: str, top_k: int):

        return retrieve(
            question=question,
            embedder=embedder,
            store=vector_store,
            top_k=top_k,
            doc_id=payload.document_id,
        )

    result = answer_question(
        question=payload.question,
        session_id=payload.document_id,
        retrieve_fn=retrieve_fn,
        llm_client=llm_client,
    )

    return AskResponse(**result)


# ============================================================
# DOCUMENTS
# ============================================================

@router.get("/documents", response_model=ListDocumentsResponse)
def list_documents():

    documents = []

    for doc_id, meta in document_registry.items():

        documents.append(
            DocumentInfo(
                document_id=doc_id,
                filename=meta["filename"],
                chunks_count=meta["chunks_count"],
                upload_timestamp=str(meta["upload_timestamp"]),
            )
        )

    return ListDocumentsResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(d.chunks_count for d in documents),
    )


# ============================================================
# DELETE
# ============================================================

@router.delete("/documents/{document_id}",
               response_model=DeleteDocumentResponse)
def delete_document(document_id: str):

    if document_id not in document_registry:

        raise HTTPException(
          status_code=404,
           detail="Document not found",
       )

    # CRITICAL FIX — delete from vector store (Qdrant + FAISS + RAM)
    vector_store.delete_document(document_id)

    # remove from registry
    del document_registry[document_id]

    save_document_registry()

    return DeleteDocumentResponse(
    document_id=document_id,
    message="Deleted",
    success=True,
)


# ============================================================
# METRICS
# ============================================================

@router.get("/metrics")
def get_metrics():

    return metrics_tracker.get_metrics()


