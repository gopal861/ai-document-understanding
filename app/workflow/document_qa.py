# app/workflow/document_qa.py

from typing import Callable, Dict, List
from app.config import SIMILARITY_THRESHOLD, TOP_K
import logging

# IMPORTANT: Use centralized production prompt builder
from app.prompts.prompt_builder import build_document_prompt

logger = logging.getLogger(__name__)


def answer_question(
    question: str,
    session_id: str,
    retrieve_fn: Callable,
    llm_client,
    top_k: int = TOP_K,
) -> Dict:
    """
    Core document QA workflow.

    Execution steps:
    1. Retrieve relevant context chunks
    2. Validate similarity threshold
    3. Build grounded prompt using centralized prompt builder
    4. Generate answer via LLM client
    5. Return structured response with confidence and safety flags
    """

    # Step 1: Retrieve context
    context_chunks = retrieve_fn(question, top_k)

    logger.info(
    "retrieval_complete",
    extra={
        "chunks_retrieved": len(context_chunks),
        "top_score": context_chunks[0]["similarity_score"] if context_chunks else None,
        "doc_id": session_id,
    },
)
    # Step 2: Handle no context found
    if not context_chunks:
        return {
            "answer": "I don't have enough information in the document to answer this.",
            "document_id": session_id,
            "confidence_score": 0.0,
            "refused": True,
            "sources_used": 0,
            "reasoning": "No relevant context found",
        }

    # Step 3: Similarity threshold safety check
    top_score = context_chunks[0]["similarity_score"]

    if top_score < SIMILARITY_THRESHOLD:
        return {
            "answer": "Not enough confident information in the document.",
            "document_id": session_id,
            "confidence_score": top_score,
            "refused": True,
            "sources_used": len(context_chunks),
            "reasoning": f"Similarity {top_score:.3f} below threshold {SIMILARITY_THRESHOLD}",
        }

    # Step 4: Build grounded prompt and generate answer
    try:
        prompt = build_document_prompt(
            question=question,
            context_chunks=context_chunks,
            model_type="cloud"
        )

        answer = llm_client.generate(prompt)

    except Exception as e:
        logger.error(
            "LLM generation failed",
            extra={"error": str(e)},
            exc_info=True
        )

        return {
            "answer": "LLM error occurred while generating the answer.",
            "document_id": session_id,
            "confidence_score": top_score,
            "refused": True,
            "sources_used": len(context_chunks),
            "reasoning": f"LLM failure: {str(e)}",
        }

    # Step 5: Return successful response
    return {
        "answer": answer,
        "document_id": session_id,
        "confidence_score": top_score,
        "refused": False,
        "sources_used": len(context_chunks),
        "reasoning": None,
    }
