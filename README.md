# Enterprise AI Research Agent

A Retrieval-Augmented Generation (RAG) application that lets users upload
PDF documents and ask natural-language questions, receiving grounded
answers with citations back to the source material.

Built for the MODUS Enterprise AI Build Challenge.

**Live app:** https://enterprise-ai-research-agent-82luoten4nbizdbsn5wuu9.streamlit.app
**Backend API:** https://enterprise-ai-research-agent-qlpf.onrender.com

> Note: this is a demo deployment on free-tier hosting with a single
> shared knowledge base — please don't upload sensitive documents. The
> backend may take up to a minute to respond on first load if it has
> been idle (Render free tier cold start).

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
                              │
                        Response Cache
```

1. **Ingestion** (`app/ingestion.py`) — extracts text from PDFs, splits
   into overlapping word-based chunks.
2. **Embeddings** (`app/embeddings.py`) — converts chunks into vectors
   using `fastembed` (ONNX Runtime based), stores them in a FAISS index
   for fast similarity search.
3. **Generation** (`app/generation.py`) — retrieves the most relevant
   chunks for a query and asks Groq's LLM to answer using only that
   context, reducing hallucination.
4. **Backend** (`app/main.py`) — FastAPI server exposing `/upload` and
   `/query` endpoints, with an in-memory response cache to avoid
   redundant LLM calls for repeated questions.
5. **Frontend** (`app/ui.py`) — Streamlit interface for uploading docs
   and asking questions.

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
cp .env.example .env
# edit .env and add your key from console.groq.com

# 4. Run the backend (Terminal 1)
uvicorn app.main:app --reload --port 8000

# 5. Run the frontend (Terminal 2)
streamlit run app/ui.py
```

Then open the Streamlit URL shown in your terminal, upload a PDF, and
start asking questions.

## Design Decisions

- **Word-based chunking with overlap**: avoids cutting sentences in half
  at chunk boundaries, so context isn't lost.
- **fastembed (ONNX-based) embeddings**: chosen over `sentence-transformers`
  (PyTorch-based) specifically to reduce memory footprint — the original
  PyTorch approach exceeded Render's free-tier 512MB memory limit and
  caused deployment crashes (exit 137/OOM). Switching to a lighter,
  ONNX-based library solved this without any loss of retrieval quality.
- **FAISS IndexFlatIP (exact search)**: fine for hundreds/thousands of
  chunks. At larger scale (100k+ records), this would be swapped for an
  approximate index like `IndexIVFFlat` to trade a little accuracy for
  much faster search.
- **Low temperature (0.2) generation**: keeps answers factual and
  grounded rather than creative, appropriate for enterprise use.
- **Source citation on every answer**: every response shows which
  document(s) it drew from, and the UI has an expandable section showing
  the actual retrieved passages — this is the explainability layer.
- **In-memory response caching**: repeated identical questions are served
  from cache instead of triggering a new LLM call, reducing cost and
  latency. The cache key includes the current indexed chunk count, so
  uploading a new document automatically invalidates stale cached answers.
- **Persisted FAISS index**: the index is saved to disk after every
  upload, so restarting the server doesn't lose previously indexed
  documents (within the lifetime of the hosting instance's disk).

## Scaling Considerations (100 → 100,000 records)

- Swap `IndexFlatIP` for `IndexIVFFlat` or a managed vector DB (Pinecone/
  Weaviate) for approximate nearest-neighbor search at scale.
- Move from synchronous upload processing to an async job queue so large
  document batches don't block the API.
- Replace the in-memory cache with a shared cache (e.g. Redis) so it
  survives restarts and works across multiple backend instances.
- Move uploaded files and the FAISS index to persistent cloud storage
  (e.g. S3) rather than local disk, since free-tier hosting disk is
  ephemeral and wiped on redeploy.

## Deployment

- **Backend**: deployed on Render (Python 3.11.9, pinned via
  `PYTHON_VERSION` env var — same fix used on an earlier FastAPI
  project — to avoid Render defaulting to an incompatible Python
  version).
- **Frontend**: deployed on Streamlit Community Cloud, pointing
  `API_URL` in `app/ui.py` at the live Render backend URL.
- Backend and frontend are intentionally split across two platforms,
  each optimized for its own workload, rather than hosting both on one
  service.

## Known Limitations (Demo Scope)

- Single shared knowledge base — all users of the live demo share the
  same uploaded documents and vector index (not multi-tenant).
- In-memory cache and FAISS index are lost if the Render instance
  restarts or redeploys (free-tier disk/memory is not permanent).
- No authentication — the app is open to anyone with the link.

These are acceptable trade-offs for a challenge demo, and are the first
things I'd address for a real production deployment (per-user/tenant
isolation, persistent cloud storage, and access control).

