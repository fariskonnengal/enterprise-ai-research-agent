"""
embeddings.py
Converts text chunks into vector embeddings and manages a FAISS index
for fast similarity search.

Uses fastembed (ONNX Runtime based) instead of sentence-transformers
(PyTorch based) to keep memory usage low enough for free-tier hosting
like Render's 512MB plan.
"""

import faiss
import numpy as np
import pickle
from fastembed import TextEmbedding
from app.ingestion import Chunk

# BAAI/bge-small-en-v1.5: small, fast, ONNX-based embedding model.
# Comparable quality to all-MiniLM-L6-v2, much lighter to run.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class VectorStore:
    def __init__(self):
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        self.index = None          # FAISS index (holds the vectors)
        self.chunks: list[Chunk] = []  # parallel list: chunks[i] matches vector i

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Helper: runs fastembed and returns a proper float32 NumPy array."""
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings, dtype="float32")

    def build_index(self, chunks: list[Chunk]):
        """Embeds all chunks and builds a searchable FAISS index from scratch."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        embeddings = self._embed(texts)

        # Normalize vectors so inner product = cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # IP = inner product (fast, exact search)
        self.index.add(embeddings)

    def add_chunks(self, chunks: list[Chunk]):
        """Adds new chunks to an existing index (for adding new documents later)."""
        if self.index is None:
            self.build_index(chunks)
            return

        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Returns the top_k most relevant chunks for a query, along with
        their similarity scores (higher = more relevant, max 1.0).
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        query_vec = self._embed([query])
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if fewer than top_k results exist
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    def save(self, path: str):
        """Persist the index and chunks to disk so we don't re-embed on restart."""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.chunks", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.chunks", "rb") as f:
            self.chunks = pickle.load(f)
