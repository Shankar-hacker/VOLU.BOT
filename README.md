# VOLU.BOT — PDF Document Chatbot

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![LangChain](https://img.shields.io/badge/LLM-LangChain-121212)
![Gemini](https://img.shields.io/badge/Gemini-Pro-4285F4)
![FAISS](https://img.shields.io/badge/Vector-FAISS-green)
![ChromaDB](https://img.shields.io/badge/Vector-ChromaDB-FF6B6B)

## What it does

Upload any PDF and ask natural language questions — the system retrieves exact passages and cites the precise page number in every answer. Manage multiple chat sessions to organize conversations by topic or document.

## Key Features

- **Multi-PDF Support**: Upload and index multiple PDFs (up to 200MB each)
- **Chat Session Management**: Create, switch between, and delete multiple chat sessions
- **Document Summary**: AI-generated summary of uploaded documents
- **Page Citations**: Every answer includes exact page references
- **Neumorphic UI**: Modern dark theme with glowing effects

## Tech Stack

| Component      | Tool                      | Why                              |
|----------------|---------------------------|----------------------------------|
| LLM            | gemini-pro (configurable) | Stable, widely available, lower demand |
| Embeddings     | gemini-embedding-001      | Free tier; 768-dim output (MRL) via Google AI |
| PDF Parsing    | PyMuPDF + pdfplumber      | Text + table extraction          |
| Vector DB (1)  | FAISS                     | In-memory, no server needed      |
| Vector DB (n)  | ChromaDB (SQLite3)        | Metadata filtering by filename, persistent storage |
| Framework      | LangChain + langchain-google-genai | Abstracts retrieval loop |
| Frontend       | Streamlit + custom CSS    | Neumorphism dark UI with red accents |
| Sessions       | Streamlit session_state   | Multi-conversation memory management |

## Setup and Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd volu-bot
```

### 2. Create virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Google AI API Key

- Get your API key from [Google AI Studio](https://aistudio.google.com/)
- Copy `.env.example` to `.env`

```bash
cp .env.example .env
```

### 5. Configure environment variables

Edit the `.env` file with your credentials:

```env
GOOGLE_API_KEY=your_google_api_key_here
CHUNK_SIZE=400
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
MAX_FILE_SIZE_MB=200
GEMINI_MODEL=gemini-pro
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_OUTPUT_DIMENSIONALITY=768
REQUESTS_PER_MINUTE=15
```

### 6. Run tests (optional)

```bash
python -m pytest test_validation.py -v
```

All 5 tests should pass.

### 7. Run the application locally

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```text
volu-bot/
├── app.py                    ← Streamlit UI with neumorphism dark theme
├── config.py                 ← All constants + check_keys()
├── requirements.txt
├── .env.example
├── test_validation.py        ← 5 pytest unit tests
├── README.md
└── utils/
    ├── __init__.py
    ├── document_loader.py    ← PyMuPDF + pdfplumber extraction
    ├── chunker.py            ← Token-aware 400/50 chunking
    ├── embedder.py           ← Google embedding-001 + FAISS/ChromaDB index
    ├── retriever.py          ← Semantic search + Gemini answer generation
    └── validator.py          ← Query + file validation
```

## How the RAG pipeline works

1. **Upload** → PyMuPDF extracts per-page text; pdfplumber adds tables into the same page text.
2. **Chunk** → 400-token chunks, 50-token overlap, page-boundary-safe, with metadata.
3. **Embed** → Google `gemini-embedding-001` (768-dim by default via MRL), batched every 20 with rate-limit sleep.
4. **Index** → FAISS (single PDF) or ChromaDB (multi-PDF with filename metadata).
5. **Query** → embed query → top-5 retrieval → Gemini Pro generates the answer.
6. **Cite** → answers include `[Page X]` plus expandable chunk previews.
7. **Sessions** → Each chat session maintains independent conversation history; switch between sessions to organize different topics or documents.

## Chat Session Management

- **Create Sessions**: Click "➕ New Chat Session" to start a fresh conversation
- **Switch Sessions**: Use the dropdown to switch between existing sessions
- **Session Persistence**: Each session maintains its own conversation history
- **Delete Sessions**: Remove unwanted sessions (requires at least 1 session to remain)
- **Export**: Export current session's chat history as a text file

## Deployment to Streamlit Cloud

### 1. Push your code to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository, branch, and `app.py`
5. Click "Advanced settings" and add your secrets:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
CHUNK_SIZE = "400"
CHUNK_OVERLAP = "50"
TOP_K_RESULTS = "5"
MAX_FILE_SIZE_MB = "200"
GEMINI_MODEL = "gemini-pro"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIMENSIONALITY = "768"
REQUESTS_PER_MINUTE = "15"
```

6. Click "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`

## Rate Limit Notes

Free tier (Google AI Studio): about 15 requests/minute (varies by account). Control embedding pace via `REQUESTS_PER_MINUTE=15` in `.env` — `utils/embedder.py` sleeps between batches.

Large PDFs (100+ pages) may take several minutes to index on a strict free tier. With billing enabled, you can raise `REQUESTS_PER_MINUTE`.

## Troubleshooting

### Windows: `pip install` fails with `WinError 32` (file in use)

Close other Python/IDE processes using the same `venv`, temporarily pause antivirus real-time scan for the project folder, then run:

```bash
pip install -r requirements.txt
```

If `sentence-transformers` (large PyTorch download) causes issues, install the rest first, then:

```bash
pip install sentence-transformers
```

### Error: `google.api_core.exceptions.ResourceExhausted`

Reduce `REQUESTS_PER_MINUTE` in `.env` (e.g. to `10`) and wait a minute before retrying.

### Error: FAISS dimension mismatch

Rebuild the index after changing `EMBEDDING_OUTPUT_DIMENSIONALITY` or the embedding model; old indexes are not compatible.

### Error: `models/embedding-001` or `text-embedding-004` not found (404)

Those models were removed from the Gemini API. Use:

```env
EMBEDDING_MODEL=gemini-embedding-001
```

The app maps old IDs automatically if you still have `embedding-001` in `.env`.

### Error: `ModuleNotFoundError: No module named 'fitz'`

Run:

```bash
pip install pymupdf
```

(The import name is `fitz`)

### Error: ChromaDB collection already exists

The app deletes and recreates the collection on each re-index using a stable persist directory in your session.

### Error: Streamlit `KeyError` in `session_state`

`app.py` uses `setdefault` for all session keys to avoid missing defaults on rerun.

## Evaluation Metrics (project goals)

- **Answer faithfulness:** % of answers grounded in retrieved chunks (target >90%)
- **Citation accuracy:** % of answers with correct page numbers (target >85%)
- **Query latency:** average response time in seconds (target <4s on free tier)
- **Multi-PDF routing precision:** correct document routing rate (target >80%)

## Future Improvements

- OCR for scanned PDFs using Tesseract
- PDF annotation export (save Q&A as PDF comments)
- Collaborative Q&A mode (multiple users)
- Document diff tool (compare two PDF versions)
- Fine-tune embedding model on domain-specific corpora

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
