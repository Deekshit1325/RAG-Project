from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text, chunk_size=None, overlap=None):
    """
    Splits text into overlapping chunks.
    
    Why chunk? Because LLMs have token limits and embedding models work better
    on smaller pieces of text. The overlap makes sure we don't cut a sentence
    in half and lose context at the boundary.
    
    Returns list of strings.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # try to break at a sentence or newline instead of mid-word
        if end < text_len:
            # look backwards from the end for a good break point
            breakpoint = _find_break_point(text, start, end)
            if breakpoint > start:
                end = breakpoint

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # move forward by (chunk_size - overlap)
        start = start + chunk_size - overlap
        # but don't go backwards
        if start <= (end - chunk_size):
            start = end

    return chunks


def _find_break_point(text, start, end):
    """
    Look for a natural break point (newline, period, etc) near the end
    of the chunk so we don't split mid-sentence.
    """
    # search in the last 20% of the chunk for a break
    search_start = max(start, end - int((end - start) * 0.2))
    segment = text[search_start:end]

    # prefer newlines > periods > spaces
    for char in ['\n', '. ', '? ', '! ']:
        idx = segment.rfind(char)
        if idx != -1:
            return search_start + idx + len(char)

    # fallback: just break at a space
    idx = segment.rfind(' ')
    if idx != -1:
        return search_start + idx + 1

    return end  # no good break point found, just cut


def chunk_document(pages):
    """
    Takes the output of document_loader (list of page dicts) and chunks each page.
    Returns a flat list of chunk dicts with metadata preserved.
    
    Each chunk looks like:
    {"text": "chunk text...", "page": 1, "source": "file.pdf", "chunk_index": 0}
    """
    all_chunks = []
    chunk_counter = 0

    for page_data in pages:
        text = page_data["text"]
        page = page_data["page"]
        source = page_data["source"]

        text_chunks = chunk_text(text)

        for chunk in text_chunks:
            all_chunks.append({
                "text": chunk,
                "page": page,
                "source": source,
                "chunk_index": chunk_counter
            })
            chunk_counter += 1

    return all_chunks
