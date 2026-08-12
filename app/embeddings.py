"""
embeddings.py
Converts text chunks into vector embeddings and manages a FAISS index
for fast similarity search.
"""

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from app.ingestion import Chunk

# all-MiniLM-L6-v2: small (80MB), fast, and good enough quality for
# most retrieval tasks. Runs on CPU with no GPU needed.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None          # FAISS index (holds the vectors)
        self.chunks: list[Chunk] = []  # parallel list: chunks[i] matches vector i

    def build_index(self, chunks: list[Chunk]):
        """Embeds all chunks and builds a searchable FAISS index from scratch."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

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
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
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

        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec, dtype="float32")
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
