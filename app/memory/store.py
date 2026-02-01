## app/memory/store.py
import faiss
import numpy as np
from typing import List, Dict, Optional

class VectorStore:
    """
    Multi-document vector store with similarity scoring.
    
    Stores embeddings from multiple documents and returns similarity scores
    for retrieval-based refusal logic.
    """
    
    def __init__(self, dim: int):
        """
        Initialize FAISS index.
        
        Args:
            dim: Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        # L2 distance index (lower distance = more similar)
        self._index = faiss.IndexFlatL2(dim)
        
        # Store chunks with metadata
        self._chunks: List[Dict] = []
    
    def add(self, embeddings: np.ndarray, chunks: List[str], doc_id: str) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            embeddings: Numpy array of shape (n_chunks, dim)
            chunks: List of text chunks
            doc_id: Document identifier
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings")
        
        # Add embeddings to FAISS index
        self._index.add(embeddings.astype('float32'))
        
        # Store chunks with metadata
        start_idx = len(self._chunks)
        for i, chunk in enumerate(chunks):
            self._chunks.append({
                "text": chunk,
                "doc_id": doc_id,
                "chunk_idx": i,
                "global_idx": start_idx + i
            })
    
    def query(
        self, 
        embedding: np.ndarray, 
        top_k: int = 5,
        doc_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve top-k most similar chunks with similarity scores.
        
        Args:
            embedding: Query embedding of shape (1, dim)
            top_k: Number of results to return
            doc_id: Optional document ID to filter results
        
        Returns:
            List of dicts with keys: text, doc_id, chunk_idx, similarity_score
        """
        if self._index.ntotal == 0:
            return []
        
        # Search FAISS index (returns L2 distances)
        distances, indices = self._index.search(embedding.astype('float32'), min(top_k * 3, self._index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            
            chunk_data = self._chunks[idx]
            
            # Filter by document ID if specified
            if doc_id and chunk_data["doc_id"] != doc_id:
                continue
            
            # Convert L2 distance to similarity score (0-1 range)
            # Lower distance = higher similarity
            # Use inverse distance with normalization
            similarity_score = 1 / (1 + dist)
            
            results.append({
                "text": chunk_data["text"],
                "doc_id": chunk_data["doc_id"],
                "chunk_idx": chunk_data["chunk_idx"],
                "similarity_score": float(similarity_score)
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """
        Get all chunks for a specific document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of text chunks
        """
        return [
            chunk["text"] 
            for chunk in self._chunks 
            if chunk["doc_id"] == doc_id
        ]
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists in store."""
        return any(chunk["doc_id"] == doc_id for chunk in self._chunks)
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        docs = set(chunk["doc_id"] for chunk in self._chunks)
        
        return {
            "total_chunks": len(self._chunks),
            "total_documents": len(docs),
            "total_vectors": self._index.ntotal,
            "document_ids": list(docs)
        }

