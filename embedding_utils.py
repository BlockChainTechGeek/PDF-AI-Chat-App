import numpy as np
import faiss
import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import streamlit as st
from functools import lru_cache


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """Load the sentence transformer model for embeddings."""
    return SentenceTransformer("hkunlp/instructor-xl")


def create_or_load_index(
    dimension: int, index_path: str = "faiss_index.npy"
) -> faiss.IndexFlatL2:
    """Create a new FAISS index or load an existing one."""
    if os.path.exists(index_path):
        try:
            index_data = np.load(index_path, allow_pickle=True).item()
            return faiss.deserialize_index(index_data)
        except Exception:
            pass
    return faiss.IndexFlatL2(dimension)


def save_index(index: faiss.IndexFlatL2, index_path: str = "faiss_index.npy"):
    """Save the FAISS index to a file."""
    try:
        index_data = faiss.serialize_index(index)
        np.save(index_path, index_data)
    except Exception as e:
        st.error(f"Error saving index: {str(e)}")


@lru_cache(maxsize=100)
def get_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    """Get the embedding for the given text using the provided model."""
    return model.encode(text)


@st.cache_data
def get_cached_embeddings(
    _index: faiss.IndexFlatL2, chunks: List[str], embedding_model: SentenceTransformer
) -> np.ndarray:
    """Get cached embeddings for the given chunks."""
    cached_embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk, embedding_model)
        cached_embeddings.append(embedding)
    return np.array(cached_embeddings)


@st.cache_data
def add_chunks_to_index(
    _index: faiss.IndexFlatL2, chunks: List[str], _embedding_model: SentenceTransformer
) -> Optional[faiss.IndexFlatL2]:
    """Add the given chunks to the FAISS index."""
    try:
        embeddings = _embedding_model.encode(
            chunks, batch_size=32, show_progress_bar=True
        )
        _index.add(embeddings.astype("float32"))
        return _index
    except Exception as e:
        st.error(f"An error occurred while adding chunks to the index: {str(e)}")
        return None


def add_conversation_to_index(
    index: faiss.IndexFlatL2,
    question: str,
    answer: str,
    embedding_model: SentenceTransformer,
):
    """Add a conversation (question and answer) to the FAISS index."""
    conversation = f"Q: {question}\nA: {answer}"
    embedding = embedding_model.encode([conversation])[0]
    index.add(np.array([embedding]).astype("float32"))


@st.cache_data
def similarity_search_cached(
    query: str,
    _index: faiss.IndexFlatL2,
    chunks: List[str],
    _embedding_model: SentenceTransformer,
    k: int = 2,
) -> List[str]:
    """Perform a similarity search for the given query using the FAISS index."""
    query_embedding = get_embedding(query, _embedding_model)
    cached_embeddings = get_cached_embeddings(_index, chunks, _embedding_model)
    D, I = _index.search(query_embedding.reshape(1, -1).astype("float32"), k)
    return [chunks[i] for i in I[0]]
