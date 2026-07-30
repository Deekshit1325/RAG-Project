import chromadb
from backend.config import CHROMA_DIR, TOP_K
from backend.embeddings import generate_embeddings

# persistent client — data survives restarts
_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )
    return _collection


def add_chunks(chunks):
    """
    Store chunks in ChromaDB. Each chunk needs a unique ID.
    
    chunks: list of dicts with keys: text, page, source, chunk_index
    """
    collection = _get_collection()

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    texts_to_embed = [c["text"] for c in chunks]
    emb_vectors = generate_embeddings(texts_to_embed)

    for i, chunk in enumerate(chunks):
        # make IDs unique per source+chunk so re-uploading same file overwrites
        chunk_id = f"{chunk['source']}__chunk_{chunk['chunk_index']}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": str(chunk["page"]),  # chroma metadata values must be strings
        })
        embeddings.append(emb_vectors[i])

    # upsert so re-uploading a file updates instead of duplicating
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(ids)


def search(query, top_k=None):
    """
    Semantic search — find the most relevant chunks for a query.
    
    Returns list of dicts:
    [{"text": "...", "source": "file.pdf", "page": "3", "score": 0.85}, ...]
    """
    if top_k is None:
        top_k = TOP_K

    collection = _get_collection()

    # check if collection is empty
    if collection.count() == 0:
        return []

    query_embedding = generate_embeddings([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    # unpack chroma's nested format into something simpler
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            # chroma returns distances, lower = more similar for cosine
            "score": round(1 - results["distances"][0][i], 4)
        })

    return hits


def get_all_documents():
    """Returns list of unique document names that have been indexed."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    
    all_meta = collection.get(include=["metadatas"])
    sources = set()
    for m in all_meta["metadatas"]:
        sources.add(m["source"])
    return sorted(list(sources))


def delete_document(source_name):
    """Remove all chunks belonging to a specific document."""
    collection = _get_collection()
    # get all IDs for this source
    results = collection.get(
        where={"source": source_name},
        include=[]
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])
    return len(results["ids"])
