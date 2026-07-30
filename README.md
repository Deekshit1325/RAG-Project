# Enterprise Multi-Document RAG Assistant

A Retrieval-Augmented Generation system that lets you upload documents (PDFs, Word docs, PowerPoints, Excel files, plain text) and ask questions about them. It finds the most relevant parts of your documents using semantic search and generates answers with citations showing exactly where the info came from.

Built as a final year CSE project.

## How It Works

1. **Upload** — You upload documents through the web UI
2. **Parse & Chunk** — The system extracts text from each file type and splits it into smaller overlapping chunks
3. **Embed & Store** — Each chunk gets converted into a vector embedding (using sentence-transformers) and stored in ChromaDB
4. **Query** — When you ask a question, it gets embedded too, and we find the most similar chunks using cosine similarity
5. **Generate** — The relevant chunks are sent to an LLM (Gemini/OpenAI/Anthropic) along with your question, and it generates an answer with citations

## Tech Stack

- **Backend**: Python, FastAPI
- **Document Parsing**: pdfplumber, python-docx, python-pptx, openpyxl
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB (local, persistent)
- **LLM**: Supports Google Gemini, OpenAI, and Anthropic (configurable)
- **Frontend**: Streamlit

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd rag
pip install -r requirements.txt
```

### 2. Configure your API key

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env` and set:
- `LLM_PROVIDER` to `gemini`, `openai`, or `anthropic`
- The corresponding API key

### 3. Run the backend

```bash
uvicorn backend.app:app --reload --port 8000
```

### 4. Run the frontend (in a separate terminal)

```bash
streamlit run frontend/app.py
```

The app should open at `http://localhost:8501`

## Features

- Multi-file upload (PDF, DOCX, PPTX, XLSX, TXT)
- Text extraction with page/slide/sheet-level metadata
- Semantic search using cosine similarity
- Answers grounded in your documents with source citations
- Swappable LLM provider (change one line in `.env`)
- Document management (view and delete indexed docs)
- Chat history within a session

## Project Structure

```
rag/
├── backend/
│   ├── app.py              # FastAPI endpoints
│   ├── document_loader.py  # Text extraction per file type
│   ├── chunker.py          # Text chunking with overlap
│   ├── embeddings.py       # Sentence-transformer embeddings
│   ├── vector_store.py     # ChromaDB operations
│   ├── llm.py              # LLM API calls (multi-provider)
│   └── config.py           # Settings and env vars
├── frontend/
│   └── app.py              # Streamlit UI
├── data/uploads/           # Temporary file storage
├── chroma_db/              # Vector DB storage
├── requirements.txt
├── .env.example
└── README.md
```

## Known Limitations

- DOCX files don't have real page numbers, so citations just say "Page 1"
- Excel extraction is basic — complex formatting, merged cells, or formulas might not parse well
- No authentication — anyone with access to the URL can use it
- Chat history is session-only (lost on page refresh)
- Chunking is character-based, not semantic — a more sophisticated chunker could improve results

## Future Improvements

- [ ] Add support for CSV files
- [ ] Hybrid search (keyword + semantic)
- [ ] Better chunking strategy (e.g., by headings/sections)
- [ ] Persistent chat history (save to DB)
- [ ] Re-ranking retrieved chunks before sending to LLM
- [ ] Support for scanned PDFs using OCR (Tesseract)
- [ ] Streaming LLM responses

## License

MIT — do whatever you want with it.
