# Enterprise AI Research Agent

A Retrieval-Augmented Generation (RAG) application that lets users upload
PDF documents and ask natural-language questions, receiving grounded
answers with citations back to the source material.

Built for the MODUS Enterprise AI Build Challenge.

## Architecture

```
User Question
     │
     ▼
Streamlit UI  ──────►  FastAPI Backend
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              Ingestion   Retrieval  Generation
              (PDF→chunks) (FAISS)   (Groq LLM)
```

1. **Ingestion** (`app/ingestion.py`) — extracts text from PDFs, splits
   into overlapping word-based chunks.
2. **Embeddings** (`app/embeddings.py`) — converts chunks into vectors
   using `sentence-transformers`, stores them in a FAISS index for fast
   similarity search.
3. **Generation** (`app/generation.py`) — retrieves the most relevant
   chunks for a query and asks Groq's LLM to answer using only that
   context, reducing hallucination.
4. **Backend** (`app/main.py`) — FastAPI server exposing `/upload` and
   `/query` endpoints.
5. **Frontend** (`app/ui.py`) — Streamlit interface for uploading docs
   and asking questions.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
cp .env.example .env
# edit .env and add your key from console.groq.com

# 3. Run the backend (Terminal 1)
uvicorn app.main:app --reload --port 8000

# 4. Run the frontend (Terminal 2)
streamlit run app/ui.py
```

Then open the Streamlit URL shown in your terminal, upload a PDF, and
start asking questions.

## Design Decisions

- **Word-based chunking with overlap**: avoids cutting sentences in half
  at chunk boundaries, so context isn't lost.
- **all-MiniLM-L6-v2 embeddings**: small (80MB), fast, no GPU required —
  good balance of speed and retrieval quality for this scale.
- **FAISS IndexFlatIP (exact search)**: fine for hundreds/thousands of
  chunks. At larger scale (100k+ records), this would be swapped for an
  approximate index like `IndexIVFFlat` to trade a little accuracy for
  much faster search.
- **Low temperature (0.2) generation**: keeps answers factual and
  grounded rather than creative, appropriate for enterprise use.
- **Source citation on every answer**: every response shows which
  document(s) it drew from, and the UI has an expandable section showing
  the actual retrieved passages — this is the explainability layer.
- **Persisted FAISS index**: the index is saved to disk after every
  upload, so restarting the server doesn't lose previously indexed
  documents.

## Scaling Considerations (100 → 100,000 records)

- Swap `IndexFlatIP` for `IndexIVFFlat` or a managed vector DB (Pinecone/
  Weaviate) for approximate nearest-neighbor search at scale.
- Move from synchronous upload processing to an async job queue so large
  document batches don't block the API.
- Add caching for frequent queries to reduce redundant LLM calls.

## Deployment

Deploy the FastAPI backend to Render (same approach as the Plant Disease
Detection API), and the Streamlit app either alongside it or on Streamlit
Community Cloud, pointing `API_URL` in `ui.py` to your deployed backend.
