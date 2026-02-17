from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import uuid
import shutil
import logging
import time
import threading

from datetime import datetime
from pathlib import Path
from typing import Dict

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
# GLOBAL SINGLETONS (ARCHITECTURE CONTRACT)
# ============================================================

embedder = Embedder()

vector_store = VectorStore(
    dim=embedder.get_dimension()
)

llm_client = MultiModelLLMClient()


# document metadata registry
document_registry: Dict[str, dict] = {}


# ============================================================
# INGESTION LOCK (CRITICAL FOR SAFETY)
# Prevents concurrent writes to FAISS index
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

    logger.info(
        "Health check",
        extra={
            "documents": len(stats["documents"]),
            "chunks": stats["total_chunks"],
            "vectors": stats["total_vectors"],
        },
    )

    return HealthResponse(
        status="healthy",
        total_documents=len(stats["documents"]),
        total_chunks=stats["total_chunks"],
        total_vectors=stats["total_vectors"],
    )


# ============================================================
# UPLOAD (SAFE INGESTION WITH LOCK)
# ============================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
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

        # ====================================================
        # LOAD TEXT
        # ====================================================

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


        if not text or not text.strip():

            raise HTTPException(
                status_code=400,
                detail="No text extracted",
            )


        # ====================================================
        # CHUNK
        # ====================================================

        chunks = chunk_text(text)

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail="No chunks created",
            )


        # ====================================================
        # EMBED
        # ====================================================

        embeddings = embedder.embed(chunks)


        # ====================================================
        # STORE (LOCK PROTECTED)
        # ====================================================

        with ingestion_lock:

            vector_store.add(
                embeddings=embeddings,
                chunks=chunks,
                doc_id=document_id,
            )


        # ====================================================
        # REGISTER
        # ====================================================

        document_registry[document_id] = {

            "filename": filename,

            "chunks_count": len(chunks),

            "upload_timestamp": datetime.utcnow(),
        }


        latency = time.time() - start_time


        logger.info(
            "Document ingestion complete",
            extra={
                "doc_id": document_id,
                "chunks": len(chunks),
                "latency_seconds": latency,
            },
        )


        return UploadResponse(
            document_id=document_id,
            filename=filename,
            chunks_created=len(chunks),
        )


    except HTTPException:

        raise


    except Exception as e:

        logger.error(
            "Document ingestion failed",
            extra={
                "doc_id": document_id,
                "error": str(e),
            },
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# ASK (SAFE — READ ONLY, NO LOCK NEEDED)
# ============================================================

@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):

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
# LIST DOCUMENTS
# ============================================================

@router.get("/documents", response_model=ListDocumentsResponse)
def list_documents():

    documents = [

        DocumentInfo(
            document_id=doc_id,
            filename=meta["filename"],
            chunks_count=meta["chunks_count"],
            upload_timestamp=meta["upload_timestamp"],
        )

        for doc_id, meta in document_registry.items()

    ]

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

    logger.info(
        "Document deleted",
        extra={"doc_id": document_id},
    )

    return DeleteDocumentResponse(
        document_id=document_id,
        message="Deleted",
        success=True,
    )


# ============================================================
# MODEL STATUS
# ============================================================

@router.get("/model-status")
def model_status():

    return llm_client.get_usage_stats()
