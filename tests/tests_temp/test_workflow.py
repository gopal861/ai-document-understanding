# tests/test_workflow.py
import pytest
from app.workflow.document_qa import answer_question, build_prompt, SIMILARITY_THRESHOLD
from app.config import TOP_K


class TestPromptBuilding:
    """Test prompt construction logic."""
    
    def test_build_prompt_with_single_chunk(self):
        """Prompt should include context with confidence score."""
        chunks = [
            {
                "text": "This is a test document about AI systems.",
                "similarity_score": 0.85,
                "doc_id": "doc_123",
                "chunk_idx": 0
            }
        ]
        
        prompt = build_prompt("What is this about?", chunks)
        
        assert "This is a test document about AI systems" in prompt
        assert "0.85" in prompt  # Confidence score
        assert "What is this about?" in prompt
        assert "ONLY the context" in prompt.upper() or "only" in prompt.lower()
    
    def test_build_prompt_with_multiple_chunks(self):
        """Prompt should format multiple chunks correctly."""
        chunks = [
            {"text": "Chunk one content", "similarity_score": 0.90, "doc_id": "doc_1", "chunk_idx": 0},
            {"text": "Chunk two content", "similarity_score": 0.75, "doc_id": "doc_1", "chunk_idx": 1},
            {"text": "Chunk three content", "similarity_score": 0.70, "doc_id": "doc_1", "chunk_idx": 2}
        ]
        
        prompt = build_prompt("Test question?", chunks)
        
        assert "Chunk one content" in prompt
        assert "Chunk two content" in prompt
        assert "Chunk three content" in prompt
        assert "Context 1" in prompt or "[Context 1" in prompt
        assert "Context 2" in prompt or "[Context 2" in prompt
        assert "Context 3" in prompt or "[Context 3" in prompt
    
    def test_build_prompt_includes_guardrails(self):
        """Prompt should include instructions to prevent hallucination."""
        chunks = [{"text": "Test content", "similarity_score": 0.80, "doc_id": "doc_1", "chunk_idx": 0}]
        
        prompt = build_prompt("Question?", chunks)
        
        # Should contain anti-hallucination instructions
        prompt_lower = prompt.lower()
        assert "only" in prompt_lower or "not" in prompt_lower
        assert "context" in prompt_lower


class TestSimilarityThresholdRefusal:
    """Test the core refusal logic based on similarity thresholds."""
    
    def test_refuse_when_no_chunks_retrieved(self):
        """Should refuse if retrieval returns no chunks."""
        def mock_retrieve(question, top_k):
            return []  # No chunks found
        
        mock_llm = None  # Won't be called
        
        result = answer_question(
            question="What is this about?",
            document_id="doc_123",
            retrieve_fn=mock_retrieve,
            llm_client=mock_llm,
            top_k=5
        )
        
        assert result["refused"] is True
        assert result["confidence_score"] == 0.0
        assert result["sources_used"] == 0
        assert "don't have" in result["answer"].lower() or "no information" in result["answer"].lower()
    
    def test_refuse_when_similarity_below_threshold(self):
        """Should refuse if top similarity score is below threshold."""
        def mock_retrieve(question, top_k):
            return [
                {
                    "text": "Some vaguely related content",
                    "similarity_score": 0.45,  # Below 0.65 threshold
                    "doc_id": "doc_123",
                    "chunk_idx": 0
                }
            ]
        
        mock_llm = None  # Won't be called
        
        result = answer_question(
            question="What is this about?",
            document_id="doc_123",
            retrieve_fn=mock_retrieve,
            llm_client=mock_llm,
            top_k=5
        )
        
        assert result["refused"] is True
        assert result["confidence_score"] == 0.45
        assert "confidence" in result["answer"].lower() or "threshold" in result["answer"].lower()
        assert SIMILARITY_THRESHOLD in result["reasoning"] or "0.65" in result["reasoning"]
    
    def test_refuse_at_exact_threshold_boundary(self):
        """Should refuse if score equals threshold (edge case)."""
        def mock_retrieve(question, top_k):
            return [
                {
                    "text": "Content at boundary",
                    "similarity_score": SIMILARITY_THRESHOLD - 0.01,  # Just below
                    "doc_id": "doc_123",
                    "chunk_idx": 0
                }
            ]
        
        result = answer_question(
            question="Test?",
            document_id="doc_123",
            retrieve_fn=mock_retrieve,
            llm_client=None,
            top_k=5
        )
        
        assert result["refused"] is True
    
    def test_answer_when_similarity_above_threshold(self):
        """Should attempt to answer if similarity is above threshold."""
        def mock_retrieve(question, top_k):
            return [
                {
                    "text": "Highly relevant content about the topic",
                    "similarity_score": 0.85,  # Above 0.65 threshold
                    "doc_id": "doc_123",
                    "chunk_idx": 0
                }
            ]
        
        class MockLLM:
            def generate(self, prompt):
                return "This is the answer based on the context."
        
        result = answer_question(
            question="What is the topic?",
            document_id="doc_123",
            retrieve_fn=mock_retrieve,
            llm_client=MockLLM(),
            top_k=5
        )
        
        assert result["refused"] is False
        assert result["confidence_score"] == 0.85
        assert result["sources_used"] == 1
        assert len(result["answer"]) > 0
        assert result["reasoning"] is None  # No refusal reasoning
    
    def test_threshold_value_is_065(self):
        """Verify threshold is set to 0.65 as claimed in resume."""
        assert SIMILARITY_THRESHOLD == 0.65


class TestRetrievalDepth:
    """Test top-k retrieval configuration."""
    
    def test_retrieval_calls_with_correct_top_k(self):
        """Verify retrieve_fn is called with correct top_k."""
        retrieved_top_k = None
        
        def mock_retrieve(question, top_k):
            nonlocal retrieved_top_k
            retrieved_top_k = top_k
            return [
                {"text": "content", "similarity_score": 0.80, "doc_id": "doc_1", "chunk_idx": 0}
            ]
        
        class MockLLM:
            def generate(self, prompt):
                return "Answer"
        
        answer_question(
            question="Test?",
            document_id="doc_1",
            retrieve_fn=mock_retrieve,
            llm_client=MockLLM(),
            top_k=5
        )
        
        assert retrieved_top_k == 5
    
    def test_default_top_k_matches_config(self):
        """Verify default top_k matches configuration."""
        from app.workflow.document_qa import TOP_K as workflow_top_k
        from app.config import TOP_K as config_top_k
        
        assert workflow_top_k == config_top_k
        assert workflow_top_k == 5  # Resume claims "top-3 to top-5"


class TestLLMGeneration:
    """Test LLM generation behavior."""
    
    def test_llm_receives_correct_prompt(self):
        """Verify LLM is called with properly formatted prompt."""
        received_prompt = None
        
        def mock_retrieve(question, top_k):
            return [
                {"text": "Test content", "similarity_score": 0.80, "doc_id": "doc_1", "chunk_idx": 0}
            ]
        
        class MockLLM:
            def generate(self, prompt):
                nonlocal received_prompt
                received_prompt = prompt
                return "Generated answer"
        
        answer_question(
            question="What is this?",
            document_id="doc_1",
            retrieve_fn=mock_retrieve,
            llm_client=MockLLM()
        )
        
        assert received_prompt is not None
        assert "Test content" in received_prompt
        assert "What is this?" in received_prompt
    
    def test_llm_failure_returns_error_response(self):
        """If LLM fails, should return refused response."""
        def mock_retrieve(question, top_k):
            return [
                {"text": "Content", "similarity_score": 0.85, "doc_id": "doc_1", "chunk_idx": 0}
            ]
        
        class BrokenLLM:
            def generate(self, prompt):
                raise Exception("API call failed")
        
        result = answer_question(
            question="Test?",
            document_id="doc_1",
            retrieve_fn=mock_retrieve,
            llm_client=BrokenLLM()
        )
        
        assert result["refused"] is True
        assert "error" in result["answer"].lower() or "encountered" in result["answer"].lower()
        assert "LLM generation failed" in result["reasoning"]


class TestResponseStructure:
    """Test that responses have correct structure."""
    
    def test_refused_response_structure(self):
        """Refused response should have all required fields."""
        def mock_retrieve(question, top_k):
            return []
        
        result = answer_question(
            question="Test?",
            document_id="doc_1",
            retrieve_fn=mock_retrieve,
            llm_client=None
        )
        
        # Check all required fields
        assert "answer" in result
        assert "document_id" in result
        assert "confidence_score" in result
        assert "refused" in result
        assert "sources_used" in result
        assert "reasoning" in result
        
        # Check types
        assert isinstance(result["answer"], str)
        assert isinstance(result["document_id"], str)
        assert isinstance(result["confidence_score"], float)
        assert isinstance(result["refused"], bool)
        asser