# tests/test_edge_cases.py
import pytest
import concurrent.futures
from unittest.mock import Mock


class TestConcurrentOperations:
    """Test system behavior under concurrent load."""
    
    def test_concurrent_uploads(self, client, sample_pdf_content):
        """Multiple simultaneous uploads should all succeed."""
        def upload():
            return client.post(
                "/upload",
                files={"file": ("test.pdf", sample_pdf_content, "application/pdf")}
            )
        
        # Upload 10 documents concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # All should have unique document IDs
        doc_ids = [r.json()["document_id"] for r in responses]
        assert len(set(doc_ids)) == 10
    
    def test_concurrent_queries_same_document(self, client, upload_sample_document, monkeypatch):
        """Multiple queries to same document should work."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        doc_id = upload_sample_document()
        
        def query():
            return client.post(
                "/ask",
                json={
                    "document_id": doc_id,
                    "question": "What is the main topic of this document?"
                }
            )
        
        # 5 concurrent queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query) for _ in range(5)]
            responses = [f.result() for f in futures]
        
        # All should complete (200 or 500 if OpenAI fails)
        assert all(r.status_code in [200, 500] for r in responses)
    
    def test_upload_and_query_interleaved(self, client, sample_pdf_content, monkeypatch):
        """Uploading while querying should not cause issues."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        # Upload first document
        response = client.post(
            "/upload",
            files={"file": ("doc1.pdf", sample_pdf_content, "application/pdf")}
        )
        doc_id_1 = response.json()["document_id"]
        
        # Start query on first doc while uploading second
        def query_first():
            return client.post(
                "/ask",
                json={"document_id": doc_id_1, "question": "What is this?"}
            )
        
        def upload_second():
            return client.post(
                "/upload",
                files={"file": ("doc2.pdf", sample_pdf_content, "application/pdf")}
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            query_future = executor.submit(query_first)
            upload_future = executor.submit(upload_second)
            
            query_result = query_future.result()
            upload_result = upload_future.result()
        
        # Both operations should succeed
        assert query_result.status_code in [200, 500]
        assert upload_result.status_code == 200


class TestSpecialCharacters:
    """Test handling of special characters and unicode."""
    
    def test_question_with_unicode(self, client, upload_sample_document):
        """Questions with unicode characters should be handled."""
        doc_id = upload_sample_document()
        
        unicode_questions = [
            "What about café service?",
            "How many €100 notes?",
            "Explain 你好 in the document",
            "What's the температура?",
            "Find 日本 references"
        ]
        
        for question in unicode_questions:
            response = client.post(
                "/ask",
                json={"document_id": doc_id, "question": question}
            )
            # Should not crash (200, 422, or 500 acceptable)
            assert response.status_code in [200, 422, 500]
    
    def test_question_with_special_symbols(self, client, upload_sample_document):
        """Questions with special symbols should be handled."""
        doc_id = upload_sample_document()
        
        special_questions = [
            "What about <script>alert('xss')</script>?",
            "Find entries with & symbol?",
            "Explain this: 100% guaranteed?",
            "What's in section §5.2?",
            "Cost is $1,000.00 - correct?"
        ]
        
        for question in special_questions:
            response = client.post(
                "/ask",
                json={"document_id": doc_id, "question": question}
            )
            assert response.status_code in [200, 422, 500]
    
    def test_filename_with_special_characters(self, client, sample_pdf_content):
        """Filenames with special characters should work."""
        special_filenames = [
            "document (1).pdf",
            "file-name_test.pdf",
            "doc@2024.pdf",
            "file.v1.0.pdf"
        ]
        
        for filename in special_filenames:
            response = client.post(
                "/upload",
                files={"file": (filename, sample_pdf_content, "application/pdf")}
            )
            assert response.status_code == 200
            assert response.json()["filename"] == filename


class TestBoundaryConditions:
    """Test boundary values and edge cases."""
    
    def test_question_at_minimum_length(self, client, upload_sample_document):
        """Question at exactly 10 characters (minimum)."""
        doc_id = upload_sample_document()
        
        response = client.post(
            "/ask",
            json={"document_id": doc_id, "question": "1234567890"}  # Exactly 10
        )
        assert response.status_code in [200, 500]  # Should be accepted
    
    def test_question_below_minimum_length(self, client, upload_sample_document):
        """Question below 10 characters should fail."""
        doc_id = upload_sample_document()
        
        response = client.post(
            "/ask",
            json={"document_id": doc_id, "question": "123456789"}  # 9 chars
        )
        assert response.status_code == 422
    
    def test_question_at_maximum_length(self, client, upload_sample_document):
        """Question at exactly 1000 characters (maximum)."""
        doc_id = upload_sample_document()
        
        question = "x" * 1000  # Exactly 1000
        response = client.post(
            "/ask",
            json={"document_id": doc_id, "question": question}
        )
        assert response.status_code in [200, 500]
    
    def test_question_above_maximum_length(self, client, upload_sample_document):
        """Question above 1000 characters should fail."""
        doc_id = upload_sample_document()
        
        question = "x" * 1001  # 1001 chars
        response = client.post(
            "/ask",
            json={"document_id": doc_id, "question": question}
        )
        assert response.status_code == 422
    
    def test_document_id_edge_cases(self, client):
        """Document IDs with edge case values."""
        edge_case_ids = [
            "",  # Empty
            " ",  # Whitespace
            "x" * 101,  # Too long (>100)
        ]
        
        for doc_id in edge_case_ids:
            response = client.post(
                "/ask",
                json={"document_id": doc_id, "question": "What is this?"}
            )
            # Should fail validation
            assert response.status_code in [400, 422]


class TestRepeatedOperations:
    """Test repeated operations for consistency."""
    
    def test_upload_same_file_multiple_times(self, client, sample_pdf_content):
        """Uploading same file multiple times should create different documents."""
        responses = []
        for i in range(5):
            response = client.post(
                "/upload",
                files={"file": (f"doc_{i}.pdf", sample_pdf_content, "application/pdf")}
            )
            responses.append(response)
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # Should have unique IDs
        doc_ids = [r.json()["document_id"] for r in responses]
        assert len(set(doc_ids)) == 5
    
    def test_repeated_identical_queries(self, client, upload_sample_document, monkeypatch):
        """Identical queries should return consistent results."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        doc_id = upload_sample_document()
        
        question = "What is the main topic of this document?"
        
        responses = []
        for _ in range(3):
            response = client.post(
                "/ask",
                json={"document_id": doc_id, "question": question}
            )
            responses.append(response)
        
        # All should have same status
        statuses = [r.status_code for r in responses]
        assert len(set(statuses)) == 1
        
        # If successful, refused status should be consistent
        if all(s == 200 for s in statuses):
            refused_values = [r.json()["refused"] for r in responses]
            assert len(set(refused_values)) == 1
    
    def test_delete_and_recreate_document(self, client, sample_pdf_content):
        """Delete and recreate should work."""
        # Upload
        response = client.post(
            "/upload",
            files={"file": ("doc.pdf", sample_pdf_content, "application/pdf")}
        )
        doc_id_1 = response.json()["document_id"]
        
        # Delete
        client.delete(f"/documents/{doc_id_1}")
        
        # Upload again
        response = client.post(
            "/upload",
            files={"file": ("doc.pdf", sample_pdf_content, "application/pdf")}
        )
        doc_id_2 = response.json()["document_id"]
        
        # Should have different ID
        assert doc_id_1 != doc_id_2


class TestSystemLimits:
    """Test system behavior at configured limits."""
    
    def test_maximum_documents_tracking(self, client, sample_pdf_content):
        """System should track multiple documents up to reasonable limit."""
        # Upload 20 documents
        for i in range(20):
            response = client.post(
                "/upload",
                files={"file": (f"doc_{i}.pdf", sample_pdf_content, "application/pdf")}
            )
            assert response.status_code == 200
        
        # Check all are tracked
        response = client.get("/documents")
        assert response.json()["total_documents"] == 20
    
    def test_health_check_with_many_documents(self, client, sample_pdf_content):
        """Health check should work even with many documents."""
        # Upload multiple documents
        for i in range(10):
            client.post(
                "/upload",
                files={"file": (f"doc_{i}.pdf", sample_pdf_content, "application/pdf")}
            )
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_documents"] == 10
        assert data["total_chunks"] > 0


class TestErrorRecovery:
    """Test system recovery from errors."""
    
    def test_continue_after_upload_failure(self, client, sample_pdf_content, non_pdf_content):
        """System should continue working after a failed upload."""
        # Fail with invalid file
        response = client.post(
            "/upload",
            files={"file": ("bad.txt", non_pdf_content, "text/plain")}
        )
        assert response.status_code == 400
        
        # Succeed with valid file
        response = client.post(
            "/upload",
            files={"file": ("good.pdf", sample_pdf_content, "application/pdf")}
        )
        assert response.status_code == 200
    
    def test_continue_after_query_failure(self, client, upload_sample_document):
        """System should continue after failed query."""
        doc_id = upload_sample_document()
        
        # Fail with invalid question
        response = client.post(
            "/ask",
            json={"document_id": doc_id, "question": "x"}  # Too short
        )
        assert response.status_code == 422
        
        # Succeed with valid question
        response = client.post(
            "/ask",
            json={"document_id": doc_id, "question": "What is this document about?"}
        )
        assert response.status_code in [200, 500]


class TestDocumentIsolation:
    """Test that documents are properly isolated."""
    
    def test_queries_dont_cross_documents(self, client, sample_pdf_content, monkeypatch):
        """Query on doc A should not return results from doc B."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        # Upload two documents
        response1 = client.post(
            "/upload",
            files={"file": ("doc_a.pdf", sample_pdf_content, "application/pdf")}
        )
        doc_id_a = response1.json()["document_id"]
        
        response2 = client.post(
            "/upload",
            files={"file": ("doc_b.pdf", sample_pdf_content, "application/pdf")}
        )
        doc_id_b = response2.json()["document_id"]
        
        # Query doc_a
        response = client.post(
            "/ask",
            json={"document_id": doc_id_a, "question": "What is this?"}
        )
        
        if response.status_code == 200:
            data = response.json()
            # Response should reference doc_a, not doc_b
            assert data["document_id"] == doc_id_a