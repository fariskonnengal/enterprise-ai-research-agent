"""
main.py
FastAPI backend exposing three endpoints:
  POST /upload  -> ingest a new document into the knowledge base
  POST /query   -> ask a question, get a grounded answer with sources
  GET  /health  -> check the service is running
"""

import os
import shutil
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingestion import process_document
from app.embeddings import VectorStore
from app.generation import generate_answer

app = FastAPI(title="Enterprise AI Research Agent")

UPLOAD_DIR = "uploads"
INDEX_PATH = "data/index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Single shared vector store in memory for this demo.
# In a multi-user production system, this would be per-tenant / per-workspace.
vector_store = VectorStore()

# Simple in-memory cache: maps a cache key -> the full query response.
# Avoids calling the LLM again for a question we've already answered,
# as long as the indexed documents haven't changed since then.
query_cache: dict[str, dict] = {}


def make_cache_key(question: str, top_k: int) -> str:
    """
    Builds a cache key from the question, top_k, and how many chunks are
    currently indexed. Including chunk count means the cache automatically
    goes stale (a fresh key) whenever new documents are uploaded, so we
    never serve an answer based on an outdated knowledge base.
    """
    raw = f"{question.strip().lower()}|{top_k}|{len(vector_store.chunks)}"
    return hashlib.sha256(raw.encode()).hexdigest()

# On startup, try to load a previously saved index so uploaded documents
# aren't lost every time the server restarts (e.g. on Render redeploys).
if os.path.exists(f"{INDEX_PATH}.index"):
    try:
        vector_store.load(INDEX_PATH)
        print(f"Loaded existing index with {len(vector_store.chunks)} chunks.")
    except Exception as e:
        print(f"Could not load existing index, starting fresh: {e}")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health_check():
    return {"status": "ok", "documents_indexed": len(vector_store.chunks)}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accepts a PDF, extracts + chunks + embeds it, and adds it to the index."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = process_document(file_path, source_name=file.filename)

    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    vector_store.add_chunks(chunks)
    vector_store.save(INDEX_PATH)  # persist immediately so a restart doesn't lose this document
    query_cache.clear()  # new document changes what "correct" answers look like, so invalidate cache

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "total_chunks_in_index": len(vector_store.chunks),
    }


@app.post("/query")
def query_documents(request: QueryRequest):
    """Retrieves relevant chunks for the question and generates a grounded answer.
    Checks an in-memory cache first to avoid redundant LLM calls for repeated
    questions against the same set of indexed documents."""
    if len(vector_store.chunks) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a document first.")

    cache_key = make_cache_key(request.question, request.top_k)
    if cache_key in query_cache:
        cached_response = query_cache[cache_key]
        return {**cached_response, "cached": True}

    results = vector_store.search(request.question, top_k=request.top_k)
    response = generate_answer(request.question, results)

    result = {
        "question": request.question,
        "answer": response["answer"],
        "sources": response["sources"],
        "retrieved_chunks": [
            {"source": c.source, "score": round(score, 3), "preview": c.text[:150]}
            for c, score in results
        ],
    }

    query_cache[cache_key] = result  # store for next time
    return {**result, "cached": False}
