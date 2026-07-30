import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import UPLOAD_DIR
from backend.document_loader import load_document, SUPPORTED_EXTENSIONS
from backend.chunker import chunk_document
from backend.vector_store import add_chunks, search, get_all_documents, delete_document
from backend.llm import generate_answer

app = FastAPI(title="RAG Assistant API")

# allow streamlit (or anything on localhost) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload one or more documents. They get parsed, chunked, and indexed."""
    results = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "skipped",
                "reason": f"Unsupported type. Supported: {SUPPORTED_EXTENSIONS}"
            })
            continue

        # save file to disk
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        with open(filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # extract text
        pages = load_document(filepath)
        if not pages:
            results.append({
                "filename": file.filename,
                "status": "skipped",
                "reason": "No text could be extracted"
            })
            continue

        # chunk it
        chunks = chunk_document(pages)

        # store in vector DB
        num_stored = add_chunks(chunks)

        results.append({
            "filename": file.filename,
            "status": "success",
            "pages_extracted": len(pages),
            "chunks_created": num_stored
        })

    return {"results": results}


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Ask a question — retrieves relevant chunks and generates an answer."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question can't be empty")

    # retrieve relevant chunks
    hits = search(req.question, top_k=req.top_k)

    if not hits:
        return QueryResponse(
            answer="No documents have been uploaded yet, or no relevant content was found.",
            sources=[]
        )

    # generate answer using LLM
    answer = generate_answer(req.question, hits)

    # format sources for the response
    sources = []
    seen = set()
    for h in hits:
        key = f"{h['source']}_p{h['page']}"
        if key not in seen:
            sources.append({
                "document": h["source"],
                "page": h["page"],
                "relevance_score": h["score"]
            })
            seen.add(key)

    return QueryResponse(answer=answer, sources=sources)


@app.get("/documents")
async def list_documents():
    """List all documents that have been uploaded and indexed."""
    docs = get_all_documents()
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents/{filename}")
async def remove_document(filename: str):
    """Remove a document from the vector store."""
    count = delete_document(filename)
    # also delete the file from uploads if it exists
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return {"deleted_chunks": count, "filename": filename}
