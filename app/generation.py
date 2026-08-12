"""
generation.py
Takes a user query + retrieved chunks, builds a grounded prompt,
and calls Groq's LLM to generate a cited, trustworthy answer.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from app.ingestion import Chunk

load_dotenv()


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Fast, free-tier friendly model on Groq
MODEL_NAME = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are an Enterprise AI Research Assistant.
Answer the user's question using ONLY the information in the provided
context. If the context doesn't contain enough information to answer
confidently, say so explicitly rather than guessing.

Always cite which source document your answer came from.
Keep answers concise and business-appropriate."""


def build_context(results: list[tuple[Chunk, float]]) -> str:
    """Formats retrieved chunks into a context block for the prompt."""
    context_parts = []
    for chunk, score in results:
        context_parts.append(
            f"[Source: {chunk.source}, relevance: {score:.2f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, results: list[tuple[Chunk, float]]) -> dict:
    """
    Generates a grounded answer using retrieved context.
    Returns the answer text plus the sources used, for transparency.
    """
    if not results:
        return {
            "answer": "I couldn't find any relevant information in the uploaded documents to answer this question.",
            "sources": [],
        }

    context = build_context(results)

    user_message = f"""Context from documents:
{context}

Question: {query}

Answer the question using only the context above. Cite sources by name."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,  # low temperature = more factual, less creative drift
        max_tokens=500,
    )

    answer_text = response.choices[0].message.content

    sources_used = list({chunk.source for chunk, _ in results})  # unique source names

    return {
        "answer": answer_text,
        "sources": sources_used,
    }
