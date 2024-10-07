import time
from functools import wraps
import streamlit as st
import psutil
from typing import List, Tuple
from llama_cpp import Llama
import hashlib
from pypdf import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
from text_processing import truncate_text
from transformers import pipeline

def timed_execution(func):
    """Decorator to measure and display the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        st.info(f"Operation took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def check_memory():
    """Check the available system memory and display a warning if it's low."""
    if psutil.virtual_memory().available < 500 * 1024 * 1024:  # 500 MB
        st.warning("Low memory. Performance may be affected.")

@st.cache_resource
def load_llm() -> Llama:
    model_path = "/Users/samuelturay/Projects/LLMProject/mistral-7b-v0.1.Q4_K_M.gguf"
    return Llama(model_path=model_path, n_ctx=2048, n_threads=4, n_gpu_layers=0)

def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def generate_answer_qa(_qa_pipeline, _question: str, _context: str) -> str:
    try:
        result = _qa_pipeline(question=_question, context=_context)
        return result["answer"]
    except Exception as e:
        st.error(f"An error occurred while generating the answer with QA Pipeline: {str(e)}")
        return ""

def generate_answer_llm(_llm: Llama, _question: str, _context: str) -> str:
    try:
        prompt = f"""Context: {_context}\n\nQuestion: {_question} Based ONLY on the information provided in the context above, please answer the question. If the information is not available in the context, say "I don't have enough information to answer that question based on the given context."\n\nAnswer:"""
        truncated_prompt = truncate_text(prompt, max_tokens=1500)
        response = _llm(truncated_prompt, max_tokens=500, stop=["Question:", "\n\n"], echo=False)
        answer = response["choices"][0]["text"].strip()
        if not answer or answer == _question or len(answer.split()) < 5:
            raise ValueError("The Generated Answer is Empty or Irrelevant")
        return answer
    except Exception as e:
        st.error(f"An error occurred while generating the answer with LLM: {str(e)}")
        return "I'm sorry, I couldn't generate an answer. Please try rephrasing your question."

@st.cache_data
def get_conversation_context(conversation_history: List[Tuple[str, str]], max_tokens: int = 200) -> str:
    """Get the context from the conversation history."""
    context = ""
    for q, a in reversed(conversation_history):
        new_entry = f"Q: {q}\nA: {a}\n\n"
        if len(context) + len(new_entry) > max_tokens:
            break
        context = new_entry + context
    return context.strip()

@st.cache_data
def get_cached_answer(question: str, context: str) -> str:
    """Retrieve a cached answer for the given question and context."""
    key = hashlib.md5((question + context).encode()).hexdigest()
    return st.session_state.get(f"cached_answer_{key}", "")

def set_cached_answer(question: str, context: str, answer: str):
    """Cache the answer for the given question and context."""
    key = hashlib.md5((question + context).encode()).hexdigest()
    st.session_state[f"cached_answer_{key}"] = answer

def get_pdf_text(pdf_docs: List[UploadedFile]) -> str:
    """Extract text from the uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        st.write(f"Processing PDF: {pdf.name}")
        try:
            pdf_reader = PdfReader(pdf)
            st.write(f"Number of pages in {pdf.name}: {len(pdf_reader.pages)}")
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    st.write(f"Warning: Empty page encountered in {pdf.name}")
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            if st.session_state.debug_mode:
                st.write(f"Debug - PDF processing error details: {traceback.format_exc()}")
    st.write(f"Total extracted text length: {len(text)} characters")
    return text
