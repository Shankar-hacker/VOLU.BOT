"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


def _normalize_embedding_model(raw: str) -> str:
    """
    Map deprecated Google embedding IDs to a model that supports embedContent.

    `models/embedding-001` and `text-embedding-004` are no longer available on
    the Gemini API; use `gemini-embedding-001` (see Google AI embeddings docs).
    """
    s = (raw or "").strip()
    if s.startswith("models/"):
        s = s[len("models/") :]
    legacy = {
        "embedding-001",
        "text-embedding-004",
        "text-embedding-005",
    }
    if s in legacy:
        return "gemini-embedding-001"
    return s or "gemini-embedding-001"


GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K: int = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_FILE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "200"))

EMBEDDING_MODEL: str = _normalize_embedding_model(
    os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
)
# Default to gemini-1.5-flash (stable, lower demand); override in .env if needed
LLM_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash").strip()

RATE_LIMIT_RPM: int = int(os.getenv("REQUESTS_PER_MINUTE", "15"))

EMBEDDING_BATCH_SIZE: int = 20

# Matryoshka output size for gemini-embedding-001 (768 keeps indexes small; max is 3072)
EMBEDDING_OUTPUT_DIMENSIONALITY: int = int(
    os.getenv("EMBEDDING_OUTPUT_DIMENSIONALITY", "768")
)

EMBEDDING_DIMENSION: int = EMBEDDING_OUTPUT_DIMENSIONALITY


def check_keys() -> bool:
    """Return True if GOOGLE_API_KEY is present and non-empty."""
    return bool(GOOGLE_API_KEY)
