"""
VOLU.BOT — Streamlit RAG chatbot with neumorphism UI.
"""

from __future__ import annotations

import functools
import html
import http.server
import logging
import os
import re
import socket
import tempfile
import threading
import urllib.parse
from datetime import datetime
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from utils import chunker, document_loader, embedder, retriever, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

COLLECTION_NAME = "pdf_oracle_docs"
PREVIEW_SERVER_HOST = "127.0.0.1"
PREVIEW_PORT_START = 8765

NEUMORPH_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@400;500;600&family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"]  {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 16px !important;
    color: #e8e8e8 !important;
  }
  .stApp {
    background-color: #1a1a1a !important;
  }
  section[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    box-shadow: 6px 6px 14px #111111, -4px -4px 10px #2a2a2a !important;
  }
  .oracle-title {
    font-family: 'Rajdhani', sans-serif !important;
    text-align: center;
    font-size: 48px !important;
    font-weight: 700 !important;
    letter-spacing: 6px !important;
    background: linear-gradient(135deg, #e63946 0%, #ff6b6b 50%, #e63946 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-shadow: 0 0 30px rgba(230,57,70,0.6), 0 0 60px rgba(230,57,70,0.4), 0 0 90px rgba(230,57,70,0.2) !important;
    filter: drop-shadow(0 0 20px rgba(230,57,70,0.5)) !important;
    margin-bottom: 0.15rem !important;
    animation: glow 3s ease-in-out infinite alternate !important;
  }
  @keyframes glow {
    from {
      filter: drop-shadow(0 0 20px rgba(230,57,70,0.5)) !important;
    }
    to {
      filter: drop-shadow(0 0 35px rgba(230,57,70,0.8)) drop-shadow(0 0 50px rgba(230,57,70,0.4)) !important;
    }
  }
  .divider-line {
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(230,57,70,0.5), transparent);
    margin: 1rem 0;
    box-shadow: 0 0 10px rgba(230,57,70,0.3);
  }
  .sidebar-divider {
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(230,57,70,0.4), transparent);
    margin: 1.5rem 0;
    box-shadow: 0 0 8px rgba(230,57,70,0.25);
  }
  .oracle-subtitle {
    text-align: center;
    font-size: 13px !important;
    letter-spacing: 4px !important;
    color: #888888 !important;
    margin-bottom: 1.25rem !important;
  }
  .neu-panel {
    background: #1f1f1f !important;
    border-radius: 14px !important;
    /*padding: 1rem 1.1rem !important;*/
    border: 1px solid rgba(40, 40, 40, 0.9) !important;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.35), -1px -1px 6px rgba(60,60,60,0.12) !important;
    margin-bottom: 1rem !important;
  }
  .section-heading-wrap {
    margin-bottom: 1.15rem !important;
    margin-top: 0.35rem !important;
  }
  .section-heading {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 20px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #0ec7ed !important;
    font-weight: 700 !important;
    display: inline-block !important;
    position: relative !important;
    padding-bottom: 10px !important;
    margin-bottom: 0 !important;
  }
  .section-heading::after {
    content: '' !important;
    position: absolute !important;
    left: 0 !important;
    bottom: 0 !important;
    height: 2px !important;
    width: 100% !important;
    border-radius: 1px !important;
    background: linear-gradient(90deg, rgba(230,57,70,0.25), rgba(230,57,70,0.55), rgba(230,57,70,0.25)) !important;
    box-shadow: 0 0 6px rgba(230,57,70,0.35) !important;
    transform: scaleX(0.12) !important;
    transform-origin: left center !important;
    transition: transform 0.55s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.45s ease, opacity 0.35s ease !important;
    opacity: 0.85 !important;
  }
 .section-heading-wrap:hover .section-heading::after {
  transform: scaleX(1) !important;
  /* New blue background gradient */
  background: linear-gradient(90deg, #0ffce8, rgba(0,255,255,0.55), #0ffce8) !important;
  /* New blue glow */
  box-shadow: 0 0 14px rgba(0,255,255,0.55), 0 0 6px rgba(0,255,255,0.35) !important;
  }
  [data-testid="stChatMessage"] h1,
  [data-testid="stChatMessage"] h2,
  [data-testid="stChatMessage"] h3 {
    font-size: 1.35rem !important;
    line-height: 1.35 !important;
    margin-top: 0.5rem !important;
  }
  div.stButton > button:first-child {
    background: #e63946 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    box-shadow: 0 0 14px rgba(230,57,70,0.4), 0 0 30px rgba(230,57,70,0.15) !important;
  }
  div.stButton > button:first-child:hover {
    box-shadow: 0 0 24px rgba(230,57,70,0.6) !important;
    color: #ffffff !important;
  }
  button[kind="secondary"] {
    background: transparent !important;
    color: #e63946 !important;
    border: 1.5px solid #e63946 !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px rgba(230,57,70,0.15) !important;
  }
  button[kind="secondary"]:hover {
    background: rgba(230,57,70,0.1) !important;
    box-shadow: 0 0 16px rgba(230,57,70,0.3) !important;
  }
  .oracle-link-btn {
    display: inline-block;
    width: 100%;
    background: #e63946;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    box-shadow: 0 0 14px rgba(230,57,70,0.4), 0 0 30px rgba(230,57,70,0.15);
    transition: box-shadow 0.25s ease, transform 0.15s ease;
  }
  .oracle-link-btn:hover {
    box-shadow: 0 0 24px rgba(230,57,70,0.6), 0 0 40px rgba(230,57,70,0.18);
    transform: translateY(-1px);
  }
  .oracle-link-note {
    margin: 0.4rem 0 0 0;
    font-size: 12px;
    color: #9a9a9a;
    letter-spacing: 0.5px;
  }
  [data-testid="stTextInput"] input,
  [data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px solid rgba(230,57,70,0.25) !important;
    color: #f0f0f0 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    line-height: 1.4 !important;
  }
  [data-testid="stChatInput"] {
    padding: 6px 4px 8px 4px !important;
  }
  [data-testid="stTextInput"] input:focus,
  [data-testid="stChatInput"] textarea:focus {
    border-color: #e63946 !important;
    box-shadow: 0 0 14px rgba(230,57,70,0.2) !important;
  }
  .citation-badge {
    display: inline-block;
    background: rgba(230,57,70,0.15);
    border: 1px solid rgba(230,57,70,0.3);
    color: #e63946;
    font-size: 11px;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px 4px 2px 0;
  }
  .status-badge {
    display: inline-block;
    font-size: 11px;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 4px 6px 4px 0;
  }
  .status-green {
    background: rgba(20,200,80,0.1);
    border: 1px solid rgba(20,200,80,0.3);
    color: #5fdc8c;
  }
  .status-red {
    background: rgba(230,57,70,0.12);
    border: 1px solid rgba(230,57,70,0.3);
    color: #e63946;
  }
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-thumb {
    background: rgba(230,57,70,0.3);
    border-radius: 4px;
  }
  [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
</style>
"""


def _init_session() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("uploaded_entries", [])
    st.session_state.setdefault("indexed", False)
    st.session_state.setdefault("summary_text", "")
    st.session_state.setdefault("index_mode", None)
    st.session_state.setdefault("faiss_index", None)
    st.session_state.setdefault("chunk_metadata", None)
    st.session_state.setdefault("chroma_collection", None)
    st.session_state.setdefault("chroma_persist_dir", None)
    st.session_state.setdefault("stats_pages", 0)
    st.session_state.setdefault("stats_chunks", 0)
    st.session_state.setdefault("viewer_selection", None)
    st.session_state.setdefault("preview_server", None)
    st.session_state.setdefault("show_summary", False)
    # Chat session management
    st.session_state.setdefault("chat_sessions", {"Session 1": []})
    st.session_state.setdefault("current_session", "Session 1")
    st.session_state.setdefault("session_counter", 1)


def _chat_export_text() -> str:
    current_session = st.session_state.get("current_session", "Session 1")
    chat_sessions = st.session_state.get("chat_sessions", {})
    current_chat = chat_sessions.get(current_session, [])
    
    lines: list[str] = [
        "VOLU.BOT — Chat export",
        f"Session: {current_session}",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
    ]
    for turn in current_chat:
        role = turn.get("role", "")
        content = turn.get("content", "")
        lines.append(f"[{role.upper()}]\n{content}\n")
        cites = turn.get("citations") or []
        if cites:
            lines.append("Citations:")
            for c in cites:
                lines.append(
                    f"  - {c.get('filename')} p.{c.get('page')}: {c.get('chunk_preview','')[:120]}"
                )
            lines.append("")
    return "\n".join(lines)


def _section_heading_html(title: str) -> str:
    safe = html.escape(title)
    return (
        f'<div class="section-heading-wrap">'
        f'<span class="section-heading">{safe}</span></div>'
    )


def _viewer_html_page() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PDF Preview</title>
  <style>
    html, body { margin:0; height:100%; background:#1a1a1a; }
    #wrap { width:100%; height:100%; }
    object, embed {
      width: 100%;
      height: 100%;
      border: none;
      background: #1a1a1a;
    }
    .fallback {
      color:#ddd;
      padding:1rem;
      font-family:sans-serif;
    }
  </style>
</head>
<body>
  <div id="wrap"></div>
  <script>
    const p = new URLSearchParams(window.location.search);
    const file = p.get("file");
    const wrap = document.getElementById("wrap");
    if (!file) {
      wrap.innerHTML = '<div class="fallback">No PDF selected.</div>';
    } else {
      const safe = file.replace(/"/g, "%22");
      wrap.innerHTML =
        '<object data="' + safe + '" type="application/pdf">' +
        '<embed src="' + safe + '" type="application/pdf" />' +
        '<div class="fallback">Could not render embedded PDF. Use Download PDF.</div>' +
        '</object>';
    }
  </script>
</body>
</html>
"""


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name)
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "document"
    ext = ".pdf" if ext.lower() != ".pdf" else ext.lower()
    return f"{stem}{ext}"


def _pick_available_port(start: int, end: int = 8795) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((PREVIEW_SERVER_HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError("Could not allocate local preview server port.")


def _ensure_preview_server() -> dict:
    existing = st.session_state.get("preview_server")
    if existing:
        return existing

    root = tempfile.mkdtemp(prefix="pdf_oracle_preview_")
    static_dir = os.path.join(root, "static")
    os.makedirs(static_dir, exist_ok=True)
    with open(os.path.join(root, "viewer.html"), "w", encoding="utf-8") as fh:
        fh.write(_viewer_html_page())

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    port = _pick_available_port(PREVIEW_PORT_START)
    handler = functools.partial(_QuietHandler, directory=root)
    server = http.server.ThreadingHTTPServer((PREVIEW_SERVER_HOST, port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    info = {
        "base_url": f"http://{PREVIEW_SERVER_HOST}:{port}",
        "static_dir": static_dir,
        "server": server,
    }
    st.session_state["preview_server"] = info
    return info


def _materialize_pdf_for_preview(filename: str, pdf_bytes: bytes) -> str:
    server_info = _ensure_preview_server()
    safe_name = _sanitize_filename(filename)
    static_path = os.path.join(server_info["static_dir"], safe_name)
    with open(static_path, "wb") as fh:
        fh.write(pdf_bytes)
    return f"/static/{urllib.parse.quote(safe_name)}"


def _pdf_iframe_html(viewer_url: str, height: int = 720) -> str:
    safe = html.escape(viewer_url, quote=True)
    return (
        f'<iframe src="{safe}" width="100%" height="{height}px" '
        'style="border:none;border-radius:12px;box-shadow:6px 6px 14px #111111,-4px -4px 10px #2a2a2a;background:#141414;"></iframe>'
    )


def main() -> None:
    st.set_page_config(
        page_title="VOLU.BOT",
        layout="wide",
        page_icon="🤖",
    )
    st.markdown(NEUMORPH_CSS, unsafe_allow_html=True)

    _init_session()

    st.markdown('<div class="oracle-title">VOLU.BOT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="oracle-subtitle">UPLOAD · INDEX · ASK — GROUNDED ANSWERS WITH PAGE CITATIONS</div>',
        unsafe_allow_html=True,
    )
    
    # Divider line after header
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

    if not config.check_keys():
        st.error(
            "Missing GOOGLE_API_KEY. Copy `.env.example` to `.env` and add your key from Google AI Studio."
        )
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.2,
    )

    with st.sidebar:
        st.markdown(_section_heading_html("Documents"), unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Add PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Max {config.MAX_FILE_MB} MB per file.",
        )

        if uploaded:
            for f in uploaded:
                if not validator.validate_file_type(f.name):
                    st.error(f"Invalid file type (not PDF): {f.name}")
                    continue
                if not validator.validate_file_size(f.size):
                    st.error(f"File too large: {f.name}")
                    continue
                exists = any(
                    e["name"] == f.name for e in st.session_state["uploaded_entries"]
                )
                if exists:
                    continue
                data = f.getvalue()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(data)
                tmp.close()
                st.session_state["uploaded_entries"].append(
                    {"name": f.name, "path": tmp.name, "bytes": data}
                )
                st.session_state["indexed"] = False

        if st.session_state["uploaded_entries"]:
            st.markdown("**Queued files**")
            to_remove: list[str] = []
            for idx, entry in enumerate(st.session_state["uploaded_entries"]):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.caption(entry["name"])
                with c2:
                    if st.button("✕", key=f"rm_{idx}_{entry['name']}", help="Remove"):
                        to_remove.append(entry["name"])
                        try:
                            os.unlink(entry["path"])
                        except OSError:
                            pass
            if to_remove:
                st.session_state["uploaded_entries"] = [
                    e
                    for e in st.session_state["uploaded_entries"]
                    if e["name"] not in to_remove
                ]
                st.session_state["indexed"] = False
                st.rerun()

        index_clicked = st.button("Index documents", type="primary")

        # Divider line after index button
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Document summary toggle button and display
        if st.session_state["indexed"] and st.session_state["summary_text"]:
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            
            if st.button("📄 View Document Summary", type="secondary", use_container_width=True, key="summary_toggle_button"):
                st.session_state["show_summary"] = not st.session_state["show_summary"]
            
            if st.session_state["show_summary"]:
                st.markdown(
                    """
                    <style>
                    .sidebar-summary-box {
                        margin-top: 1rem;
                        border: 2px solid rgba(20,200,80,0.5);
                        border-radius: 10px;
                        background: rgba(20,200,80,0.08);
                        box-shadow: 0 0 12px rgba(20,200,80,0.25);
                        padding: 1rem;
                    }
                    .sidebar-summary-box .summary-content {
                        background: rgba(10,50,20,0.3);
                        padding: 1rem;
                        border-radius: 8px;
                        color: #e8e8e8;
                        font-size: 14px;
                        line-height: 1.6;
                    }
                    .sidebar-summary-box .summary-title {
                        color: #5fdc8c;
                        font-weight: 600;
                        font-size: 15px;
                        margin-bottom: 0.75rem;
                        letter-spacing: 1px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'''
                    <div class="sidebar-summary-box">
                        <div class="summary-title">📄 DOCUMENT SUMMARY</div>
                        <div class="summary-content">{st.session_state["summary_text"]}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

        # Divider line before chat sessions
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Chat Sessions Management
        st.markdown(_section_heading_html("Chat Sessions"), unsafe_allow_html=True)
        
        # Custom styling for new session button with light red
        st.markdown(
            """
            <style>
            div[data-testid="stVerticalBlock"] > div:has(button[key="new_session_btn"]) button {
                background: #ff6b6b !important;
                color: #ffffff !important;
                border: none !important;
                font-weight: 600 !important;
                box-shadow: 0 0 14px rgba(255,107,107,0.4) !important;
            }
            div[data-testid="stVerticalBlock"] > div:has(button[key="new_session_btn"]) button:hover {
                background: #ff5252 !important;
                box-shadow: 0 0 20px rgba(255,107,107,0.6) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # New session button
        if st.button("✙ New Chat Session", type="secondary", use_container_width=True, key="new_session_btn"):
            st.session_state["session_counter"] += 1
            new_session_name = f"Session {st.session_state['session_counter']}"
            st.session_state["chat_sessions"][new_session_name] = []
            st.session_state["current_session"] = new_session_name
            st.rerun()
        
        # Session selector
        session_names = list(st.session_state["chat_sessions"].keys())
        if session_names:
            selected_session = st.selectbox(
                "Select Chat Session",
                session_names,
                index=session_names.index(st.session_state["current_session"]),
                key="session_selector"
            )
            
            if selected_session != st.session_state["current_session"]:
                st.session_state["current_session"] = selected_session
                st.rerun()
            
            # Display session info
            current_chat = st.session_state["chat_sessions"][st.session_state["current_session"]]
            msg_count = len(current_chat)
            st.caption(f"💬 {msg_count} messages in this session")
            
            # Delete session button (only if more than 1 session exists)
            if len(session_names) > 1:
                if st.button("🗑️ Delete Current Session", type="secondary", use_container_width=True):
                    del st.session_state["chat_sessions"][st.session_state["current_session"]]
                    st.session_state["current_session"] = list(st.session_state["chat_sessions"].keys())[0]
                    st.rerun()

        

    if index_clicked:
        if not st.session_state["uploaded_entries"]:
            st.warning("Upload at least one PDF before indexing.")
        else:
            with st.spinner("Indexing documents — extracting, chunking, embedding…"):
                try:
                    all_pages: list[dict] = []
                    for entry in st.session_state["uploaded_entries"]:
                        path = entry["path"]
                        document_loader.validate_pdf(path)
                        pages = document_loader.load_pdf(path)
                        tables = document_loader.extract_tables(path)
                        merged = document_loader.merge_tables_into_pages(pages, tables)
                        all_pages.extend(merged)

                    chunks = chunker.chunk_pages(
                        all_pages,
                        config.CHUNK_SIZE,
                        config.CHUNK_OVERLAP,
                    )
                    if not chunks:
                        st.error("No text could be extracted from the PDF(s).")
                    else:
                        filenames = sorted(
                            {str(c["filename"]) for c in chunks}
                        )
                        unique_count = len(filenames)

                        if unique_count == 1:
                            idx, meta = embedder.build_faiss_index(
                                chunks, config.GOOGLE_API_KEY
                            )
                            st.session_state["faiss_index"] = idx
                            st.session_state["chunk_metadata"] = meta
                            st.session_state["chroma_collection"] = None
                            st.session_state["index_mode"] = "faiss"
                        else:
                            if st.session_state["chroma_persist_dir"] is None:
                                st.session_state["chroma_persist_dir"] = (
                                    tempfile.mkdtemp(prefix="chroma_pdf_oracle_")
                                )
                            coll = embedder.build_chroma_index(
                                chunks,
                                COLLECTION_NAME,
                                config.GOOGLE_API_KEY,
                                st.session_state["chroma_persist_dir"],
                            )
                            st.session_state["chroma_collection"] = coll
                            st.session_state["faiss_index"] = None
                            st.session_state["chunk_metadata"] = None
                            st.session_state["index_mode"] = "chroma"

                        st.session_state["stats_pages"] = len(all_pages)
                        st.session_state["stats_chunks"] = len(chunks)
                        st.session_state["indexed"] = True

                        summary = document_loader.generate_summary(all_pages, llm)
                        st.session_state["summary_text"] = summary

                        names = [e["name"] for e in st.session_state["uploaded_entries"]]
                        st.session_state["viewer_selection"] = names[0]
                        st.success("Documents indexed successfully.")
                except document_loader.PDFTooLargeError as exc:
                    st.error(str(exc))
                except document_loader.InvalidPDFTypeError as exc:
                    st.error(str(exc))
                except document_loader.CorruptPDFError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    logger.exception("Indexing failed")
                    st.error(f"Indexing failed: {exc}")

    # Chat section - centered, full width with larger font
    st.markdown(
        """
        <style>
        .chat-title-center {
            text-align: center;
            margin: 0.5rem 0 1.5rem 0;
        }
        .chat-title-center .section-heading {
            font-size: 32px !important;
            letter-spacing: 5px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="chat-title-center">{_section_heading_html("Chat")}</div>',
        unsafe_allow_html=True,
    )

    exp_col, dl_col = st.columns(2)
    with exp_col:
        st.caption("Export full Q&A history as a text file.")
    with dl_col:
        st.download_button(
            label="Export chat (.txt)",
            data=_chat_export_text(),
            file_name="pdf_oracle_chat.txt",
            mime="text/plain",
            use_container_width=True,
            type="secondary",
        )

    if not st.session_state["indexed"]:
        st.warning("Index your documents in the sidebar before chatting.")
    else:
        # Get current session's chat history
        current_chat = st.session_state["chat_sessions"][st.session_state["current_session"]]
        
        for turn in current_chat:
            role = turn["role"]
            with st.chat_message(role):
                st.markdown(turn["content"])
                if role == "assistant" and turn.get("citations"):
                    with st.expander("Sources — pages & chunk previews"):
                        for c in turn["citations"]:
                            st.markdown(
                                f'<span class="citation-badge">'
                                f'{c.get("filename","")} · p.{c.get("page")}</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(c.get("chunk_preview", ""))

        user_q = st.chat_input("Ask a question about your PDFs…")
        if user_q:
            ok, err = validator.validate_query(user_q)
            if not ok:
                st.error(err)
            else:
                st.session_state["chat_sessions"][st.session_state["current_session"]].append(
                    {"role": "user", "content": user_q}
                )
                with st.chat_message("user"):
                    st.markdown(user_q)

                try:
                    filenames = [
                        e["name"] for e in st.session_state["uploaded_entries"]
                    ]
                    routed = retriever.route_query(user_q, filenames, llm)

                    if st.session_state["index_mode"] == "faiss":
                        chunks = retriever.retrieve_faiss(
                            user_q,
                            st.session_state["faiss_index"],
                            st.session_state["chunk_metadata"],
                            config.GOOGLE_API_KEY,
                            config.TOP_K,
                        )
                    else:
                        filt = None if routed == "all" else routed
                        chunks = retriever.retrieve_chroma(
                            user_q,
                            st.session_state["chroma_collection"],
                            config.GOOGLE_API_KEY,
                            config.TOP_K,
                            filename_filter=filt,
                        )

                    result = retriever.generate_answer(user_q, chunks, llm)
                    answer = result["answer"]
                    cites = result["citations"]
                except Exception as exc:
                    logger.exception("Chat pipeline failed")
                    answer = "Something went wrong while answering. Please try again."
                    cites = []

                st.session_state["chat_sessions"][st.session_state["current_session"]].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": cites,
                    }
                )
                st.rerun()


if __name__ == "__main__":
    main()
