"""
ui.py
Streamlit frontend for the Enterprise AI Research Agent.
Lets a user upload PDF documents and ask questions, seeing the
grounded answer plus which sources/chunks were used.
"""

import streamlit as st
import requests

API_URL = "https://enterprise-ai-research-agent-qlpf.onrender.com"  # change this to your deployed backend URL

st.set_page_config(page_title="Enterprise AI Research Agent", page_icon="🔎", layout="wide")

st.title("🔎 Enterprise AI Research Agent")
st.caption("Upload documents, ask questions, get grounded answers with citations.")

# ---- Sidebar: document upload ----
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Index this document"):
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(
                            f"Indexed '{result['filename']}' — "
                            f"{result['chunks_created']} chunks created "
                            f"({result['total_chunks_in_index']} total in index)."
                        )
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the backend. Is the FastAPI server running?")

    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.metric("Documents indexed (chunks)", health["documents_indexed"])
    except requests.exceptions.ConnectionError:
        st.warning("Backend not running yet.")

# ---- Main area: ask questions ----
st.header("💬 Ask a Question")
question = st.text_input("What would you like to know about your documents?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Searching documents and generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": 5},
                timeout=60,
            )
            if response.status_code == 200:
                result = response.json()

                st.subheader("Answer")
                st.write(result["answer"])

                if result["sources"]:
                    st.caption(f"📚 Sources: {', '.join(result['sources'])}")

                with st.expander("🔍 See retrieved passages (for transparency)"):
                    for chunk in result["retrieved_chunks"]:
                        st.markdown(
                            f"**{chunk['source']}** (relevance: {chunk['score']})  \n"
                            f"_{chunk['preview']}..._"
                        )
                        st.divider()
            else:
                st.error(f"Query failed: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the backend. Is the FastAPI server running?")
