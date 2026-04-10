"""Token-aware chunking with page boundaries preserved."""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _encode_len(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_page_tokens(
    text: str, chunk_size: int, overlap: int
) -> list[str]:
    """Split single-page text into token chunks with overlap; never crosses given text."""
    if not text or not text.strip():
        return []

    tokens = _ENCODING.encode(text)
    if len(tokens) <= chunk_size:
        return [_ENCODING.decode(tokens)]

    chunks: list[str] = []
    start = 0
    stride = max(1, chunk_size - overlap)

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        piece = tokens[start:end]
        chunks.append(_ENCODING.decode(piece))
        if end >= len(tokens):
            break
        start += stride

    return chunks


def chunk_pages(
    pages: list[dict], chunk_size: int, overlap: int
) -> list[dict]:
    """
    Chunk each page independently (never across page boundaries).

    Each output dict: {text, filename, page_number, chunk_index}
    """
    logger.info(
        "chunk_pages: %d pages, chunk_size=%d overlap=%d",
        len(pages),
        chunk_size,
        overlap,
    )

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    all_chunks: list[dict] = []

    for page in pages:
        filename = page["filename"]
        page_number = int(page["page_number"])
        raw_text = page.get("text") or ""
        pieces = _split_page_tokens(raw_text, chunk_size, overlap)

        for idx, piece in enumerate(pieces):
            if not piece.strip():
                continue
            all_chunks.append(
                {
                    "text": piece.strip(),
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_index": idx,
                }
            )

    logger.info("chunk_pages: produced %d chunks", len(all_chunks))
    return all_chunks
