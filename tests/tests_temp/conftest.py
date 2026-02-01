# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add app directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

@pytest.fixture
def client():
    """
    FastAPI test client.
    
    Used to make requests to the API in tests.
    """
    return TestClient(app)


@pytest.fixture
def sample_pdf_content():
    """
    Minimal valid PDF content for testing.
    
    This is a real PDF that parsers can read.
    """
    # Minimal PDF with text content
    pdf_content = b"""%PDF-1.4
1 0 obj

/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj

/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj

/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources 
/Font 
/F1 5 0 R
>>
>>
>>
endobj
4 0 obj

/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test document content) Tj
ET
endstream
endobj
5 0 obj

/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000270 00000 n 
0000000363 00000 n 
trailer

/Size 6
/Root 1 0 R
>>
startxref
441
%%EOF"""
    return pdf_content


@pytest.fixture
def sample_pdf_with_specific_content():
    """
    PDF with known content for targeted question testing.
    """
    # This would be a real PDF with specific content
    # For now, using the same minimal PDF
    return sample_pdf_content()


@pytest.fixture
def upload_sample_document(client, sample_pdf_content):
    """
    Upload a sample document and return its ID.
    
    Helper fixture that handles the upload process.
    """
    def _upload():
        response = client.post(
            "/upload",
            files={"file": ("test_document.pdf", sample_pdf_content, "application/pdf")}
        )
        assert response.status_code == 200, f"Upload failed: {response.json()}"
        return response.json()["document_id"]
    
    return _upload


@pytest.fixture
def large_pdf_content():
    """
    Generate a PDF larger than the size limit for testing.
    """
    # Create a PDF that exceeds 10MB
    large_content = b"%PDF-1.4\n" + b"x" * (11 * 1024 * 1024) + b"\n%%EOF"
    return large_content


@pytest.fixture
def empty_pdf_content():
    """
    Empty/corrupted PDF for error testing.
    """
    return b""


@pytest.fixture
def non_pdf_content():
    """
    Non-PDF file content for validation testing.
    """
    return b"This is a plain text file, not a PDF."


@pytest.fixture(autouse=True)
def reset_app_state():
    """
    Reset application state between tests.
    
    This ensures tests don't interfere with each other.
    """
    from app.api import routes
    
    # Reset global state before each test
    routes.vector_store = None
    routes.document_registry.clear()
    
    yield
    
    # Cleanup after test
    routes.vector_store = None
    routes.document_registry.clear()


@pytest.fixture
def mock_openai_response(monkeypatch):
    """
    Mock OpenAI API responses to avoid making real API calls in tests.
    
    Usage:
        def test_something(mock_openai_response):
            mock_openai_response("This is a mocked answer")
            # ... test code ...
    """
    def _mock(response_text: str):
        from unittest.mock import Mock, MagicMock
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = response_text
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        def mock_create(*args, **kwargs):
            return mock_response
        
        from app.llm import client
        monkeypatch.setattr(client.OpenAI, "chat", MagicMock())
        monkeypatch.setattr(
            client.OpenAI.return_value.chat.completions,
            "create",
            mock_create
        )
    
    return _mock