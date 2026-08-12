"""
ingestion.py
Loads documents (PDFs) and splits them into overlapping text chunks
that are small enough to embed and retrieve accurately.
"""

from pypdf import PdfReader
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str      # filename the chunk came from
    chunk_id: int     # position within the document


def extract_text_from_pdf(file_path: str) -> str:
    """Reads a PDF file and returns all its text as one string."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text += page_text + "\n"
    return full_text


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """
    Splits text into overlapping chunks.

    chunk_size: number of words per chunk (not characters — keeps chunks
                semantically coherent rather than cutting mid-sentence too often)
    overlap: number of words repeated between consecutive chunks, so context
             isn't lost at chunk boundaries
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        if chunk_str.strip():  # skip empty chunks
            chunks.append(Chunk(text=chunk_str, source=source, chunk_id=chunk_id))
            chunk_id += 1

        start += chunk_size - overlap  # move forward, but overlap with previous chunk

    return chunks


def process_document(file_path: str, source_name: str) -> list[Chunk]:
    """Full pipeline: PDF -> raw text -> list of chunks ready for embedding."""
    text = extract_text_from_pdf(file_path)
    return chunk_text(text, source=source_name)
