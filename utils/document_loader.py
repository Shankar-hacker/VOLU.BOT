"""PDF loading, table extraction, validation, and summary generation."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import fitz
import pdfplumber

import config

logger = logging.getLogger(__name__)


class PDFValidationError(Exception):
    """Base class for PDF validation failures."""

    pass


class PDFTooLargeError(PDFValidationError):
    """Raised when PDF exceeds configured maximum file size."""

    pass


class InvalidPDFTypeError(PDFValidationError):
    """Raised when the file is not a valid PDF type."""

    pass


class CorruptPDFError(PDFValidationError):
    """Raised when the PDF cannot be opened or read."""

    pass


def _sort_blocks_reading_order(blocks: list[dict]) -> list[dict]:
    """Sort text blocks in approximate reading order (top-to-bottom, left-to-right)."""

    def sort_key(b: dict) -> tuple[float, float]:
        bbox = b.get("bbox", (0, 0, 0, 0))
        y0, x0 = bbox[1], bbox[0]
        return (round(y0 / 5) * 5, x0)

    return sorted(blocks, key=sort_key)


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text per page using PyMuPDF with multi-column-aware block ordering.

    Returns list of dicts: {page_number, text, filename}
    """
    logger.info("load_pdf: opening %s", file_path)
    filename = os.path.basename(file_path)
    pages_out: list[dict] = []

    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        logger.exception("load_pdf: failed to open %s", file_path)
        raise CorruptPDFError(f"Could not open PDF: {exc}") from exc

    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_number = page_index + 1
            d = page.get_text("dict")
            blocks = d.get("blocks", [])
            text_blocks: list[str] = []

            for block in _sort_blocks_reading_order(blocks):
                if block.get("type") != 0:
                    continue
                lines_parts: list[str] = []
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if line_text:
                        lines_parts.append(line_text)
                if lines_parts:
                    text_blocks.append("\n".join(lines_parts))

            text = "\n\n".join(text_blocks).strip()
            pages_out.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "filename": filename,
                }
            )
    finally:
        doc.close()

    logger.info("load_pdf: extracted %d pages from %s", len(pages_out), filename)
    return pages_out


def extract_tables(file_path: str) -> list[dict]:
    """
    Extract tables with pdfplumber as readable text per page.

    Returns list of dicts: {page_number, table_text, filename}
    """
    logger.info("extract_tables: %s", file_path)
    filename = os.path.basename(file_path)
    tables_out: list[dict] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                try:
                    tables = page.extract_tables() or []
                except Exception as exc:
                    logger.warning(
                        "extract_tables: page %d table extract failed: %s",
                        page_number,
                        exc,
                    )
                    continue

                for table in tables:
                    if not table:
                        continue
                    lines: list[str] = []
                    for row in table:
                        cells = [str(c).strip() if c is not None else "" for c in row]
                        lines.append(" | ".join(cells))
                    table_text = "\n".join(lines).strip()
                    if table_text:
                        tables_out.append(
                            {
                                "page_number": page_number,
                                "table_text": table_text,
                                "filename": filename,
                            }
                        )
    except Exception as exc:
        logger.exception("extract_tables: pdfplumber failed on %s", file_path)
        raise CorruptPDFError(f"Table extraction failed: {exc}") from exc

    logger.info(
        "extract_tables: %d table segments from %s", len(tables_out), filename
    )
    return tables_out


def validate_pdf(file_path: str) -> bool:
    """
    Validate PDF exists, is under size limit, and opens cleanly.

    Raises PDFTooLargeError, InvalidPDFTypeError, or CorruptPDFError on failure.
    Returns True if valid.
    """
    logger.info("validate_pdf: %s", file_path)

    if not os.path.isfile(file_path):
        logger.error("validate_pdf: not a file: %s", file_path)
        raise InvalidPDFTypeError("Path is not a file.")

    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".pdf":
        logger.error("validate_pdf: wrong extension: %s", ext)
        raise InvalidPDFTypeError("File must have .pdf extension.")

    size_bytes = os.path.getsize(file_path)
    max_bytes = config.MAX_FILE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        logger.error(
            "validate_pdf: file too large: %d bytes > %d", size_bytes, max_bytes
        )
        raise PDFTooLargeError(
            f"PDF exceeds maximum size of {config.MAX_FILE_MB} MB."
        )

    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
    except Exception as exc:
        logger.exception("validate_pdf: corrupt or unreadable: %s", file_path)
        raise CorruptPDFError(f"Corrupted or unreadable PDF: {exc}") from exc

    if page_count < 1:
        logger.error("validate_pdf: zero pages: %s", file_path)
        raise CorruptPDFError("PDF has no pages.")

    logger.info("validate_pdf: OK, %d pages", page_count)
    return True


def merge_tables_into_pages(
    pages: list[dict], tables: list[dict]
) -> list[dict]:
    """Append extracted table text to matching page text in-place on copies."""
    page_map: dict[tuple[str, int], dict] = {}
    for p in pages:
        key = (p["filename"], int(p["page_number"]))
        page_map[key] = dict(p)

    for t in tables:
        key = (t["filename"], int(t["page_number"]))
        if key in page_map:
            existing = page_map[key]["text"] or ""
            addition = f"\n\n[TABLE]\n{t['table_text']}"
            page_map[key]["text"] = (existing + addition).strip()

    ordered: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for p in pages:
        key = (p["filename"], int(p["page_number"]))
        if key not in seen:
            seen.add(key)
            ordered.append(page_map[key])
    return ordered


def generate_summary(pages: list[dict], llm: Any) -> str:
    """Generate approximately 200 words summarizing the document via Gemini."""
    logger.info("generate_summary: %d page records", len(pages))

    # Keep prompt small to stay under free-tier generate_content limits during indexing.
    combined = []
    for p in pages[:8]:
        snippet = (p.get("text") or "")[:900]
        combined.append(f"--- Page {p['page_number']} ---\n{snippet}")
    body = "\n\n".join(combined)
    if len(body) > 24_000:
        body = body[:24_000]

    prompt = (
        "You are a concise technical summarizer. Read the following excerpts from a PDF "
        "and write a summary of roughly 200 words (180–220 words). Focus on main themes, "
        "entities, and purpose. Use plain prose. Do not invent facts not supported by the text.\n\n"
        f"DOCUMENT EXCERPTS:\n{body}"
    )

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            response = llm.invoke(prompt)
            text = getattr(response, "content", None) or str(response)
            logger.info("generate_summary: done, length=%d", len(text))
            return text.strip()
        except Exception as exc:
            last_err = exc
            err_s = str(exc)
            if attempt == 0 and (
                "429" in err_s
                or "RESOURCE_EXHAUSTED" in err_s
                or "quota" in err_s.lower()
            ):
                logger.warning(
                    "generate_summary: rate limited, retrying once after 45s"
                )
                time.sleep(45)
                continue
            break

    logger.exception("generate_summary: LLM failed")
    hint = (
        " Embeddings and chat use separate limits in AI Studio. If you see 429/RESOURCE_EXHAUSTED, "
        "wait ~1 minute or set GEMINI_MODEL to the Flash model your dashboard lists (e.g. gemini-2.5-flash)."
    )
    msg = f"Summary could not be generated: {last_err}"
    if last_err and (
        "429" in str(last_err) or "RESOURCE_EXHAUSTED" in str(last_err).upper()
    ):
        msg += hint
    return msg
