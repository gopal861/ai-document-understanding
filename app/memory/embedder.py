# app/memory/embedder.py
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]):
        return self._model.encode(texts, convert_to_numpy=True)
