# app/workflow/document_qa.py
from typing import Callable, Dict, List

# CRITICAL: Similarity threshold for refusal
SIMILARITY_THRESHOLD = 0.65
TOP_K = 5


def build_prompt(question: str, context_chunks: List[Dict]) -> str:
    """
    Build prompt for LLM with retrieved context.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        score = chunk.get("similarity_score", 0.0)
        text = chunk.get("text", "")
        context_parts.append(f"[Context {i}, Confidence: {score:.2f}]\n{text}")
    
    context = "\n\n".join(context_parts)

    return f"""You are a document understanding assistant.

CRITICAL RULES:
- Use ONLY the context provided below
- If the answer is not in the context, say "I don't know based on the document"
- Do NOT make assumptions or guess
- Do NOT use external knowledge
- Be precise and cite specific parts of the context

Context:
{context}

Question:
{question}

Answer (use only the context above):""".strip()


def answer_question(
    question: str,
    session_id: str,
    retrieve_fn: Callable,
    llm_client,
    top_k: int = TOP_K,
) -> Dict:
    """
    Answer a question using retrieval-augmented generation with confidence-based refusal.
    """
    context_chunks = retrieve_fn(question, top_k)
    
    if not context_chunks:
        return {
            "answer": "I don't have any information in the document to answer this question.",
            "document_id": session_id,
            "confidence_score": 0.0,
            "refused": True,
            "sources_used": 0,
            "reasoning": "No relevant context found in document"
        }
    
    top_similarity = context_chunks[0].get("similarity_score", 0.0)
    
    if top_similarity < SIMILARITY_THRESHOLD:
        return {
            "answer": f"I don't have enough confident information in the document to answer this question. The most relevant content has only {top_similarity:.2%} confidence, which is below the {SIMILARITY_THRESHOLD:.0%} threshold.",
            "document_id": session_id,
            "confidence_score": top_similarity,
            "refused": True,
            "sources_used": len(context_chunks),
            "reasoning": f"Top similarity score ({top_similarity:.3f}) below threshold ({SIMILARITY_THRESHOLD})"
        }
    
    prompt = build_prompt(question, context_chunks)
    
    try:
        answer = llm_client.generate(prompt)
    except Exception as e:
        return {
            "answer": "I encountered an error while processing your question. Please try again.",
            "document_id": session_id,
            "confidence_score": top_similarity,
            "refused": True,
            "sources_used": len(context_chunks),
            "reasoning": f"LLM generation failed: {str(e)}"
        }
    
    return {
        "answer": answer,
        "document_id": session_id,
        "confidence_score": top_similarity,
        "refused": False,
        "sources_used": len(context_chunks),
        "reasoning": None
    }