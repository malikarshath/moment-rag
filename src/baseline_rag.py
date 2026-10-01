"""Baseline semantic search RAG — plain cosine similarity, no decomposition,
no hybrid retrieval, no reranking.

Used as the comparison baseline against the full Moment RAG pipeline.

Pipeline:
    query → embed (1 vector) → cosine similarity over all chunks → top-k → LLM answer
"""
from __future__ import annotations

import time

import numpy as np
from openai import OpenAI

from .config import LLM_MODEL

BASELINE_PROMPT = """You are a helpful assistant. Answer the question using ONLY the transcript
excerpts below. Cite sources inline as [1], [2], etc.

Question: {query}

Excerpts:
{excerpts}

Answer:"""


def search(query: str, chunks: list[dict], top_k: int = 5) -> tuple[list[dict], float]:
    """Embed the session's chunks and query, then return cosine top-k."""
    from .embed import embed_texts

    t0 = time.perf_counter()
    if not chunks:
        return [], (time.perf_counter() - t0) * 1000

    vecs = embed_texts([c["text"] for c in chunks])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs = vecs / norms
    q_vec = embed_texts([query])[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    sims = vecs @ q_vec
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = [{"chunk": chunks[i], "score": float(sims[i])} for i in top_idx]
    elapsed = (time.perf_counter() - t0) * 1000
    return results, elapsed


def generate_answer(query: str, results: list[dict]) -> str:
    """Generate a plain LLM answer from the top-k retrieved chunks."""
    excerpts = "\n\n".join(
        f"[{i+1}] {r['chunk'].get('text','')[:600]}"
        for i, r in enumerate(results)
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        messages=[{"role": "user", "content": BASELINE_PROMPT.format(
            query=query, excerpts=excerpts)}]
    )
    return resp.choices[0].message.content or ""


def run(query: str, chunks: list[dict], top_k: int = 5) -> dict:
    """Run the baseline against only chunks ingested in this server session."""
    results, retrieve_ms = search(query, chunks, top_k)
    t0 = time.perf_counter()
    answer = generate_answer(query, results)
    gen_ms = (time.perf_counter() - t0) * 1000
    return {
        "method": "Baseline: single-vector cosine similarity",
        "steps": ["embed query (1 vector)", "cosine similarity over all chunks", "top-k", "LLM answer"],
        "chunks": [{"text": r["chunk"].get("text","")[:300],
                    "video_id": r["chunk"].get("video_id",""),
                    "start_ms": r["chunk"].get("start_ms", 0),
                    "score": round(r["score"], 4)} for r in results],
        "answer": answer,
        "timings_ms": {"retrieve": round(retrieve_ms, 1), "generate": round(gen_ms, 1)},
    }
