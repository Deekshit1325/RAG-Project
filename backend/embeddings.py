from sentence_transformers import SentenceTransformer

# load model once and reuse — this thing takes a second to init
# all-MiniLM-L6-v2 is small (~80MB), runs on CPU, and is surprisingly good
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_embeddings(texts):
    """
    Takes a list of strings, returns a list of embeddings (each is a list of floats).
    The model outputs 384-dimensional vectors.
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    # chromadb wants regular lists, not numpy arrays
    return embeddings.tolist()
