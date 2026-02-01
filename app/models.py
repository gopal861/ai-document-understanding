# app/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class UploadResponse(BaseModel):
    """Response after uploading a document."""
    document_id: str
    filename: str
    chunks_created: int
    message: str = "Document uploaded and indexed successfully"


class AskRequest(BaseModel):
    """Request to ask a question about a document."""
    document_id: str = Field(..., min_length=1, max_length=100)
    question: str = Field(..., min_length=10, max_length=1000)
    
    @validator('question')
    def validate_question(cls, v):
        """Ensure question is not just whitespace."""
        if not v.strip():
            raise ValueError("Question cannot be empty or only whitespace")
        return v.strip()
    
    @validator('document_id')
    def validate_document_id(cls, v):
        """Ensure document_id is not just whitespace."""
        if not v.strip():
            raise ValueError("Document ID cannot be empty")
        return v.strip()


class AskResponse(BaseModel):
    """Response after asking a question."""
    answer: str
    document_id: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    refused: bool
    sources_used: int
    reasoning: Optional[str] = None


class DocumentInfo(BaseModel):
    """Information about a stored document."""
    document_id: str
    filename: str
    chunks_count: int
    upload_timestamp: Optional[str] = None


class ListDocumentsResponse(BaseModel):
    """Response listing all documents in the system."""
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteDocumentResponse(BaseModel):
    """Response after deleting a document."""
    document_id: str
    message: str
    success: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    total_documents: int
    total_chunks: int
    total_vectors: int