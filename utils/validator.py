"""User query and upload validation helpers."""

from __future__ import annotations

import logging
import os

import config

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 1000


def validate_query(query: str) -> tuple[bool, str]:
    """Return (is_valid, error_message). Empty string error_message when valid."""
    if query is None:
        logger.warning("validate_query: None query")
        return False, "Query cannot be empty."

    stripped = query.strip()
    if not stripped:
        logger.warning("validate_query: whitespace-only")
        return False, "Query cannot be empty or whitespace only."

    if len(stripped) > MAX_QUERY_CHARS:
        logger.warning("validate_query: too long (%d)", len(stripped))
        return False, f"Query must be under {MAX_QUERY_CHARS} characters."

    return True, ""


def validate_file_type(filename: str) -> bool:
    """True if extension is .pdf (case-insensitive)."""
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    ok = ext == ".pdf"
    if not ok:
        logger.warning("validate_file_type: rejected %s", filename)
    return ok


def validate_file_size(file_size_bytes: int) -> bool:
    """True if size is within MAX_FILE_MB."""
    max_bytes = config.MAX_FILE_MB * 1024 * 1024
    ok = file_size_bytes <= max_bytes
    if not ok:
        logger.warning(
            "validate_file_size: %d bytes exceeds limit %d", file_size_bytes, max_bytes
        )
    return ok
