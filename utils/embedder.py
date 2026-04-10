"""Google embeddings, FAISS and Chroma indexing with rate limiting."""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from typing import Any

import faiss
import numpy as np
from google.api_core import exceptions as google_exceptions
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config

logger = logging.getLogger(__name__)


def _sleep_between_batches() -> None:
    delay = 60.0 / float(max(1, config.RATE_LIMIT_RPM))
    logger.debug("embedder: sleeping %.2fs between batches (RPM=%d)", delay, config.RATE_LIMIT_RPM)
    time.sleep(delay)


def embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    """
    Embed texts with Google Gemini embedding model via LangChain.

    Batches of EMBEDDING_BATCH_SIZE with rate-limit sleep between batches.
    """
    if not texts:
        return []

    logger.info("embed_texts: count=%d", len(texts))

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=config.EMBEDDING_OUTPUT_DIMENSIONALITY,
    )

    all_vectors: list[list[float]] = []
    batch_size = config.EMBEDDING_BATCH_SIZE

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch_emb = embeddings_model.embed_documents(batch)
        except google_exceptions.ResourceExhausted as exc:
            logger.error("embed_texts: ResourceExhausted: %s", exc)
            raise
        except google_exceptions.InvalidArgument as exc:
            logger.error("embed_texts: InvalidArgument: %s", exc)
            raise
        except google_exceptions.GoogleAPIError as exc:
            logger.error("embed_texts: GoogleAPIError: %s", exc)
            raise
        except Exception as exc:
            logger.exception("embed_texts: unexpected error")
            raise

        all_vectors.extend(batch_emb)
        if i + batch_size < len(texts):
            _sleep_between_batches()

    logger.info("embed_texts: completed %d vectors", len(all_vectors))
    return all_vectors


def build_faiss_index(
    chunks: list[dict], api_key: str
) -> tuple[Any, list[dict]]:
    """Build FAISS IndexFlatL2 over chunk embeddings; return (index, metadata list)."""
    logger.info("build_faiss_index: chunks=%d", len(chunks))

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, api_key)

    if not embeddings:
        raise ValueError("No embeddings produced for FAISS index.")

    dim = len(embeddings[0])
    if dim != config.EMBEDDING_DIMENSION:
        logger.warning(
            "build_faiss_index: embedding dim=%d expected %d",
            dim,
            config.EMBEDDING_DIMENSION,
        )

    xb = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(xb)

    metadata = [dict(c) for c in chunks]
    logger.info("build_faiss_index: index.ntotal=%d", index.ntotal)
    return index, metadata


def build_chroma_index(
    chunks: list[dict],
    collection_name: str,
    api_key: str,
    persist_directory: str | None = None,
) -> Any:
    """Build ChromaDB collection with filename metadata. Uses a persistent on-disk path."""
    import chromadb

    if persist_directory is None:
        persist_directory = tempfile.mkdtemp(prefix="chroma_pdf_oracle_")

    logger.info(
        "build_chroma_index: chunks=%d collection=%s path=%s",
        len(chunks),
        collection_name,
        persist_directory,
    )

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, api_key)

    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "l2"},
    )

    ids = [f"id_{uuid.uuid4().hex}" for _ in chunks]
    metadatas = [
        {
            "filename": str(c["filename"]),
            "page_number": int(c["page_number"]),
            "chunk_index": int(c["chunk_index"]),
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info("build_chroma_index: added %d documents", len(ids))
    return collection
