import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from llama_cpp import Llama

@st.cache_resource
def load_llm():
    model_path = "/Users/samuelturay/Projects/LLMProject/mistral-7b-v0.1.Q4_K_M.gguf"
    return Llama(model_path=model_path, n_ctx=2048, n_threads=4, n_gpu_layers=0)

llm = load_llm()

def tokenize(text):
    return llm.tokenize(text.encode("utf-8"))

def detokenize(tokens):
    return llm.detokenize(tokens).decode("utf-8")

def preprocess_text(text: str) -> str:
    """Preprocess the input text by removing extra whitespace and non-alphanumeric characters."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9\s\.,;!?]", "", text)
    return text

def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split the input text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def truncate_text(text: str, max_tokens: int = 1000) -> str:
    tokens = tokenize(text)
    if len(tokens) > max_tokens:
        return detokenize(tokens[:max_tokens])
    return text

@st.cache_resource
def get_tfidf_vectorizer(chunks: List[str]) -> TfidfVectorizer:
    """Get or create a TF-IDF vectorizer for the given chunks."""
    vectorizer = TfidfVectorizer()
    vectorizer.fit(chunks)
    return vectorizer

@st.cache_data
def get_tfidf_matrix(_vectorizer: TfidfVectorizer, chunks: List[str]):
    """Get the TF-IDF matrix for the given chunks."""
    return _vectorizer.transform(chunks)

def select_relevant_chunks(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Select the most relevant chunks for the given query using TF-IDF and cosine similarity."""
    vectorizer = get_tfidf_vectorizer(chunks)
    tfidf_matrix = get_tfidf_matrix(vectorizer, chunks)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_k - 1:-1]
    return [chunks[i] for i in related_docs_indices]