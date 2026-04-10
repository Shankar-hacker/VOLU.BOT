"""Retrieval (FAISS / Chroma), answer generation, and query routing."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

import config
from utils.embedder import embed_texts

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "Answer the question using only the provided document context. "
    "Always cite the exact page number at the end of your answer in this format: [Page X]. "
    "If the answer is not in the document, say: I could not find this information in the uploaded document."
)


def retrieve_faiss(
    query: str,
    index: Any,
    chunk_metadata: list[dict],
    api_key: str,
    top_k: int,
) -> list[dict]:
    """Embed query, search FAISS, return top_k chunk dicts (with text)."""
    logger.info("retrieve_faiss: top_k=%d", top_k)

    if index is None or index.ntotal == 0:
        return []

    q_vec = embed_texts([query], api_key)[0]
    q = np.array([q_vec], dtype=np.float32)
    k = min(top_k, int(index.ntotal))
    distances, indices = index.search(q, k)

    out: list[dict] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(chunk_metadata):
            continue
        out.append(dict(chunk_metadata[int(idx)]))

    logger.info("retrieve_faiss: retrieved %d chunks", len(out))
    return out


def retrieve_chroma(
    query: str,
    collection: Any,
    api_key: str,
    top_k: int,
    filename_filter: str | None = None,
) -> list[dict]:
    """Query Chroma collection; optional metadata filter by filename."""
    logger.info(
        "retrieve_chroma: top_k=%d filter=%s", top_k, filename_filter
    )

    q_vec = embed_texts([query], api_key)[0]
    where = None
    if filename_filter and filename_filter.lower() != "all":
        where = {"filename": {"$eq": filename_filter}}

    result = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs_list = result.get("documents", [[]])[0]
    meta_list = result.get("metadatas", [[]])[0]

    out: list[dict] = []
    for i, doc in enumerate(docs_list):
        meta = meta_list[i] if i < len(meta_list) else {}
        out.append(
            {
                "text": doc,
                "filename": meta.get("filename", ""),
                "page_number": int(meta.get("page_number", 0)),
                "chunk_index": int(meta.get("chunk_index", 0)),
            }
        )

    logger.info("retrieve_chroma: retrieved %d chunks", len(out))
    return out


def _build_context_block(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}] File: {c.get('filename','')} | Page: {c.get('page_number')}\n"
            f"{c.get('text','')}"
        )
    return "\n\n".join(parts)


def generate_answer(
    query: str, retrieved_chunks: list[dict], llm: Any
) -> dict:
    """
    Generate grounded answer with citations metadata.

    Returns {answer, citations: [{page, filename, chunk_preview}]}
    """
    logger.info("generate_answer: chunks=%d", len(retrieved_chunks))

    if not retrieved_chunks:
        return {
            "answer": "I could not find this information in the uploaded document.",
            "citations": [],
        }

    context = _build_context_block(retrieved_chunks)
    user_content = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{query}"
    )

    try:
        response = llm.invoke(user_content)
        answer = getattr(response, "content", None) or str(response)
    except Exception as exc:
        logger.exception("generate_answer: LLM failed")
        return {
            "answer": "Sorry, the model could not produce an answer. Please try again.",
            "citations": [],
        }

    citations: list[dict] = []
    for c in retrieved_chunks:
        preview = (c.get("text") or "")[:180].replace("\n", " ")
        if len((c.get("text") or "")) > 180:
            preview += "…"
        citations.append(
            {
                "page": int(c.get("page_number", 0)),
                "filename": str(c.get("filename", "")),
                "chunk_preview": preview,
            }
        )

    unique: dict[tuple[int, str], dict] = {}
    for cit in citations:
        key = (cit["page"], cit["filename"])
        unique[key] = cit
    citations = list(unique.values())

    logger.info("generate_answer: answer length=%d", len(answer))
    return {"answer": answer.strip(), "citations": citations}


def route_query(query: str, filenames: list[str], llm: Any) -> str:
    """
    Use Gemini to pick one filename or 'all' for cross-document questions.
    """
    if not filenames:
        return "all"
    if len(filenames) == 1:
        return filenames[0]

    logger.info("route_query: candidates=%s", filenames)

    listing = "\n".join(f"- {fn}" for fn in filenames)
    prompt = (
        "You route user questions to the correct PDF filename when applicable.\n"
        f"Available files (exact names):\n{listing}\n\n"
        "Rules:\n"
        "- If the question clearly targets a single file (by name, topic unique to one file), "
        "respond with that filename exactly as listed.\n"
        "- If the question compares files, asks about all documents, or is ambiguous, respond with exactly: all\n"
        "- Output nothing else: no punctuation, no quotes, no explanation.\n\n"
        f"USER QUESTION:\n{query.strip()}"
    )

    try:
        response = llm.invoke(prompt)
        raw = (getattr(response, "content", None) or str(response)).strip()
    except Exception as exc:
        logger.warning("route_query: LLM failed (%s), defaulting to all", exc)
        return "all"

    cleaned = raw.splitlines()[0].strip().strip('"').strip("'")
    if cleaned.lower() == "all":
        return "all"
    for fn in filenames:
        if fn == cleaned or fn.lower() == cleaned.lower():
            logger.info("route_query: routed to %s", fn)
            return fn

    match = re.search(r"[\w\-. ]+\.pdf", cleaned, re.IGNORECASE)
    if match:
        candidate = match.group(0)
        for fn in filenames:
            if fn.lower() == candidate.lower():
                return fn

    logger.info("route_query: default all (unparsed=%r)", raw[:80])
    return "all"
