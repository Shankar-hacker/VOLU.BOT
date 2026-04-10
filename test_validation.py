"""Pytest suite for PDF.ORACLE validation and core utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import config
from utils import chunker, embedder, validator


def test_imports() -> None:
    """All utils modules import without error."""
    import utils.document_loader  # noqa: F401
    import utils.chunker  # noqa: F401
    import utils.embedder  # noqa: F401
    import utils.retriever  # noqa: F401
    import utils.validator  # noqa: F401


def test_config() -> None:
    """check_keys returns bool; constants have expected types."""
    assert isinstance(config.check_keys(), bool)
    assert isinstance(config.CHUNK_SIZE, int)
    assert isinstance(config.CHUNK_OVERLAP, int)
    assert isinstance(config.TOP_K, int)
    assert isinstance(config.MAX_FILE_MB, int)
    assert isinstance(config.EMBEDDING_MODEL, str)
    assert isinstance(config.LLM_MODEL, str)
    assert isinstance(config.RATE_LIMIT_RPM, int)
    assert isinstance(config.GOOGLE_API_KEY, str)


def test_chunking() -> None:
    """chunk_pages returns chunks with required metadata keys."""
    pages = [
        {
            "page_number": 1,
            "filename": "demo.pdf",
            "text": "word " * 200,
        }
    ]
    chunks = chunker.chunk_pages(pages, chunk_size=50, overlap=10)
    assert len(chunks) >= 1
    for c in chunks:
        assert set(c.keys()) >= {"text", "filename", "page_number", "chunk_index"}
        assert c["filename"] == "demo.pdf"
        assert c["page_number"] == 1


@patch("utils.embedder.GoogleGenerativeAIEmbeddings")
def test_embedding(mock_emb_cls: MagicMock) -> None:
    """embed_texts returns 768-float vectors per chunk (embedding-001)."""
    instance = MagicMock()
    instance.embed_documents.side_effect = lambda batch: [
        [0.1] * config.EMBEDDING_DIMENSION for _ in batch
    ]
    mock_emb_cls.return_value = instance

    texts = ["hello", "world"]
    vectors = embedder.embed_texts(texts, api_key="fake-key")

    assert len(vectors) == 2
    assert all(len(v) == config.EMBEDDING_DIMENSION for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])


def test_end_to_end_validation() -> None:
    """Query and file validators behave for typical inputs."""
    ok, msg = validator.validate_query("")
    assert ok is False
    assert msg

    ok, msg = validator.validate_query("What is this doc?")
    assert ok is True
    assert msg == ""

    assert validator.validate_file_type("report.pdf") is True
    assert validator.validate_file_type("data.csv") is False
