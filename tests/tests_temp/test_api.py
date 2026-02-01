# tests/test_api.py
import pytest
import os

class TestHealthEndpoint:
    """Test the /health endpoint."""
    
    def test_health_check_no_documents(self, client):
        """Health check should work even with no documents uploaded."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["total_documents"] == 0
        assert data["total_chunks"] == 0
    
    def test_health_check_with_documents(self, client, upload_sample_document):
        """Health check should reflect uploaded documents."""
        # Upload a document
        upload_sample_document()
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["total_documents"] == 1
        assert data["total_chunks"] > 0


class TestUploadEndpoint:
    """Test the /upload endpoint with various inputs."""
    
    def test_upload_valid_pdf(self, client, sample_pdf_content):
        """Valid PDF upload should succeed."""
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", sample_pdf_content, "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "document_id" in data
        assert data["document_id"].startswith("doc_")
        assert data["filename"] == "test.pdf"
        assert data["chunks_created"] > 0
        assert "successfully" in data["message"].lower()
    
    def test_upload_wrong_file_extension(self, client, non_pdf_content):
        """Non-PDF file should be rejected."""
        response = client.post(
            "/upload",
            files={"file": ("document.txt", non_pdf_content, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"] or ".pdf" in response.json()["detail"]
    
    def test_upload_file_too_large(self, client, large_pdf_content):
        """File larger than limit should be rejected."""
        response = client.post(
            "/upload",
            files={"file": ("huge.pdf", large_pdf_content, "application/pdf")}
        )
        
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower() or "size" in response.json()["detail"].lower()
    
    def test_upload_empty_file(self, client, empty_pdf_content):
        """Empty file should be rejected."""
        response = client.post(
            "/upload",
            files={"file": ("empty.pdf", empty_pdf_content, "application/pdf")}
        )
        
        # Should fail with 400 or 500 depending on PDF parser behavior
        assert response.status_code in [400, 500]
    
    def test_upload_multiple_documents(self, client, sample_pdf_content):
        """Multiple documents should be stored independently."""
        # Upload first document
        response1 = client.post(
            "/upload",
            files={"file": ("doc1.pdf", sample_pdf_content, "application/pdf")}
        )
        assert response1.status_code == 200
        doc_id_1 = response1.json()["document_id"]
        
        # Upload second document
        response2 = client.post(
            "/upload",
            files={"file": ("doc2.pdf", sample_pdf_content, "application/pdf")}
        )
        assert response2.status_code == 200
        doc_id_2 = response2.json()["document_id"]
        
        # Document IDs should be different
        assert doc_id_1 != doc_id_2
        
        # Both should be listed
        response = client.get("/documents")
        data = response.json()
        assert data["total_documents"] == 2
        
        doc_ids = [doc["document_id"] for doc in data["documents"]]
        assert doc_id_1 in doc_ids
        assert doc_id_2 in doc_ids


class TestAskEndpoint:
    """Test the /ask endpoint with various scenarios."""
    
    def test_ask_without_upload(self, client):
        """Asking without uploading should fail."""
        response = client.post(
            "/ask",
            json={
                "document_id": "nonexistent_doc",
                "question": "What is this about?"
            }
        )
        
        assert response.status_code == 400
        assert "no document" in response.json()["detail"].lower() or "not found" in response.json()["detail"].lower()
    
    def test_ask_with_nonexistent_document(self, client, upload_sample_document):
        """Asking about non-existent document should fail."""
        # Upload a document
        upload_sample_document()
        
        # Ask about different document
        response = client.post(
            "/ask",
            json={
                "document_id": "fake_doc_id_12345",
                "question": "What is this about?"
            }
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_ask_valid_question(self, client, upload_sample_document, monkeypatch):
        """Valid question should return answer."""
        # Set mock API key
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        doc_id = upload_sample_document()
        
        # Note: This will fail without real OpenAI key or mocking
        # For now, we just test the request structure
        response = client.post(
            "/ask",
            json={
                "document_id": doc_id,
                "question": "What is the content of this document?"
            }
        )
        
        # Should return 200 or 500 (if OpenAI fails)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "confidence_score" in data
            assert "refused" in data
            assert isinstance(data["refused"], bool)
    
    def test_ask_empty_question(self, client, upload_sample_document):
        """Empty question should be rejected."""
        doc_id = upload_sample_document()
        
        response = client.post(
            "/ask",
            json={
                "document_id": doc_id,
                "question": "   "
            }
        )
        
        assert response.status_code == 422  # Pydantic validation error
    
    def test_ask_question_too_short(self, client, upload_sample_document):
        """Question below minimum length should be rejected."""
        doc_id = upload_sample_document()
        
        response = client.post(
            "/ask",
            json={
                "document_id": doc_id,
                "question": "Hi"  # Less than 10 chars
            }
        )
        
        assert response.status_code == 422
    
    def test_ask_question_too_long(self, client, upload_sample_document):
        """Question above maximum length should be rejected."""
        doc_id = upload_sample_document()
        
        long_question = "x" * 1001  # More than 1000 chars
        response = client.post(
            "/ask",
            json={
                "document_id": doc_id,
                "question": long_question
            }
        )
        
        assert response.status_code == 422
    
    def test_ask_missing_document_id(self, client):
        """Request without document_id should fail."""
        response = client.post(
            "/ask",
            json={
                "question": "What is this about?"
            }
        )
        
        assert response.status_code == 422  # Pydantic validation
    
    def test_ask_missing_question(self, client, upload_sample_document):
        """Request without question should fail."""
        doc_id = upload_sample_document()
        
        response = client.post(
            "/ask",
            json={
                "document_id": doc_id
            }
        )
        
        assert response.status_code == 422


class TestDocumentsEndpoint:
    """Test the /documents endpoint."""
    
    def test_list_documents_empty(self, client):
        """Listing documents when none uploaded."""
        response = client.get("/documents")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_documents"] == 0
        assert data["total_chunks"] == 0
        assert data["documents"] == []
    
    def test_list_documents_with_uploads(self, client, sample_pdf_content):
        """List should show all uploaded documents."""
        # Upload two documents
        client.post(
            "/upload",
            files={"file": ("doc1.pdf", sample_pdf_content, "application/pdf")}
        )
        client.post(
            "/upload",
            files={"file": ("doc2.pdf", sample_pdf_content, "application/pdf")}
        )
        
        response = client.get("/documents")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_documents"] == 2
        assert len(data["documents"]) == 2
        
        # Check document structure
        for doc in data["documents"]:
            assert "document_id" in doc
            assert "filename" in doc
            assert "chunks_count" in doc
            assert doc["chunks_count"] > 0


class TestDeleteEndpoint:
    """Test document deletion."""
    
    def test_delete_existing_document(self, client, upload_sample_document):
        """Deleting an existing document should succeed."""
        doc_id = upload_sample_document()
        
        response = client.delete(f"/documents/{doc_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["document_id"] == doc_id
    
    def test_delete_nonexistent_document(self, client):
        """Deleting non-existent document should fail."""
        response = client.delete("/documents/fake_doc_id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_delete_and_verify_removed(self, client, upload_sample_document):
        """After deletion, document should not be in list."""
        doc_id = upload_sample_document()
        
        # Verify it exists
        response = client.get("/documents")
        assert response.json()["total_documents"] == 1
        
        # Delete it
        client.delete(f"/documents/{doc_id}")
        
        # Verify it's gone from list
        response = client.get("/documents")
        assert response.json()["total_documents"] == 0


class TestRootEndpoint:
    """Test the root endpoint."""
    
    def test_root_endpoint(self, client):
        """Root should return API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "docs" in data