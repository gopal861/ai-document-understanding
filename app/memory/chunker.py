# app/memory/chunker.py
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Uses word-based chunking with overlap to preserve context across boundaries.
    
    Args:
        text: Input text to chunk
        size: Number of words per chunk (default from config)
        overlap: Number of words to overlap between chunks (default from config)
    
    Returns:
        List of text chunks
    
    Example:
        >>> text = "word1 word2 word3 word4 word5"
        >>> chunk_text(text, size=3, overlap=1)
        ['word1 word2 word3', 'word3 word4 word5']
    """
    if not text or not text.strip():
        return []
    
    if size <= 0:
        raise ValueError(f"Chunk size must be positive, got {size}")
    
    if overlap < 0:
        raise ValueError(f"Overlap cannot be negative, got {overlap}")
    
    if overlap >= size:
        raise ValueError(f"Overlap ({overlap}) must be less than chunk size ({size})")
    
    words = text.split()
    
    if not words:
        return []
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Move forward by (size - overlap) to create overlap
        start += size - overlap
        
        # Prevent infinite loop if we're at the end
        if end >= len(words):
            break
    
    return chunks
