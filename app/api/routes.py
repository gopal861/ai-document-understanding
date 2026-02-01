# app/api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid
import os
from datetime import datetime
from typing import Dict

from app.models import (
    AskRequest, 
    AskResponse, 
    UploadResponse,
    ListDocumentsResponse,
    DeleteDocumentResponse,
    DocumentInfo,
    HealthResponse
)
from app.memory.loader import load_pdf_text
from app.memory.chunker import chunk_text
from app.memory.embedder import Embedder
from app.memory.store import VectorStore
from app.memory.retriever import retrieve
from app.workflow.document_qa import answer_question
from app.llm.client import LLMClient

router = APIRouter()

# ---- GLOBAL STATE (MULTI-DOCUMENT SUPPORT) ----
embedder = Embedder()
vector_store: VectorStore = None
llm_client = LLMClient()

# Document registry: stores metadata about each document
document_registry: Dict[str, dict] = {}

# Configuration
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".pdf"]


# ----------- HELPER FUNCTIONS -----------

def generate_document_id() -> str:
    """Generate unique document ID."""
    return f"doc_{uuid.uuid4().hex[:12]}"


def validate_file_size(content: bytes) -> None:
    """Validate file size is within limits."""
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB, got: {size_mb:.2f}MB"
        )


def validate_file_extension(filename: str) -> None:
    """Validate file has allowed extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported. Got: {ext}"
        )


# ----------- ENDPOINTS -----------

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Returns system status and statistics.
    """
    if vector_store is None:
        return HealthResponse(
            status="healthy",
            total_documents=0,
            total_chunks=0,
            total_vectors=0
        )
    
    stats = vector_store.get_stats()
    return HealthResponse(
        status="healthy",
        total_documents=stats["total_documents"],
        total_chunks=stats["total_chunks"],
        total_vectors=stats["total_vectors"]
    )


@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for question answering.
    
    Creates a unique document ID and stores the document in the vector store.
    Multiple documents can be uploaded and queried independently.
    """
    global vector_store
    
    try:
        # 1. Validate file extension
        validate_file_extension(file.filename)
        
        # 2. Read and validate file size
        content = file.file.read()
        validate_file_size(content)
        
        # 3. Generate unique document ID
        doc_id = generate_document_id()
        
        # 4. Save temporary file
        temp_path = f"temp_{doc_id}.pdf"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 5. Load and chunk document
        text = load_pdf_text(temp_path)
        
        if not text.strip():
            os.remove(temp_path)
            raise HTTPException(
                status_code=400, 
                detail="Document contains no readable text"
            )
        
        chunks = chunk_text(text)
        
        if not chunks:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail="Document could not be split into chunks"
            )
        
        # 6. Generate embeddings
        embeddings = embedder.embed(chunks)
        
        # 7. Initialize vector store if first document
        if vector_store is None:
            vector_store = VectorStore(dim=embeddings.shape[1])
        
        # 8. Add to vector store with document ID
        vector_store.add(embeddings, chunks, doc_id)
        
        # 9. Store metadata in registry
        document_registry[doc_id] = {
            "filename": file.filename,
            "chunks_count": len(chunks),
            "upload_timestamp": datetime.utcnow().isoformat(),
            "size_bytes": len(content)
        }
        
        # 10. Cleanup temp file
        os.remove(temp_path)
        
        return UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            chunks_created=len(chunks)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup temp file if exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    """
    Ask a question about a specific document.
    
    Uses retrieval-augmented generation with similarity threshold refusal.
    If retrieved context has low confidence, refuses to answer.
    """
    # 1. Validate vector store exists
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload a document first."
        )
    
    # 2. Validate document exists
    if not vector_store.document_exists(payload.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{payload.document_id}' not found. Use /documents to see available documents."
        )
    
    try:
        # 3. Define retrieval function with document filtering
        def retrieve_fn(question: str, top_k: int):
            return retrieve(
                question=question,
                embedder=embedder,
                store=vector_store,
                top_k=top_k,
                doc_id=payload.document_id  # Filter by document
            )
        
        # 4. Execute workflow (includes threshold refusal logic)
       # 4. Execute workflow (includes threshold refusal logic)
        result = answer_question(
    question=payload.question,
    session_id=payload.document_id,  # ← CHANGE THIS LINE
    retrieve_fn=retrieve_fn,
    llm_client=llm_client,
)
        
        return AskResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )


@router.get("/documents", response_model=ListDocumentsResponse)
def list_documents():
    """
    List all uploaded documents.
    
    Returns metadata for each document including ID, filename, and chunk count.
    """
    if not document_registry:
        return ListDocumentsResponse(
            documents=[],
            total_documents=0,
            total_chunks=0
        )
    
    documents = [
        DocumentInfo(
            document_id=doc_id,
            filename=metadata["filename"],
            chunks_count=metadata["chunks_count"],
            upload_timestamp=metadata.get("upload_timestamp")
        )
        for doc_id, metadata in document_registry.items()
    ]
    
    total_chunks = sum(doc.chunks_count for doc in documents)
    
    return ListDocumentsResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=total_chunks
    )


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str):
    """
    Delete a document from the system.
    
    Note: Current implementation doesn't remove vectors from FAISS index
    (FAISS doesn't support deletion). In production, use a database with delete support.
    """
    if document_id not in document_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found"
        )
    
    # Remove from registry
    del document_registry[document_id]
    
    return DeleteDocumentResponse(
        document_id=document_id,
        message="Document metadata removed. Note: Vectors remain in index (FAISS limitation).",
        success=True
    )