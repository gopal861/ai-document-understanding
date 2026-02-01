# app/memory/retriever.py
from typing import List, Dict, Optional

def retrieve(
    question: str,
    embedder,
    store,
    top_k: int = 5,
    doc_id: Optional[str] = None
) -> List[Dict]:
    """
    Retrieve top-k most similar chunks with similarity scores.
    
    Args:
        question: User's question
        embedder: Embedder instance to generate query embedding
        store: VectorStore instance to search
        top_k: Number of results to return
        doc_id: Optional document ID to filter results
    
    Returns:
        List of dicts with keys: text, doc_id, chunk_idx, similarity_score
    """
    # Generate query embedding
    query_embedding = embedder.embed([question])
    
    # Search vector store (with optional document filter)
    results = store.query(
        embedding=query_embedding,
        top_k=top_k,
        doc_id=doc_id
    )
    
    return results