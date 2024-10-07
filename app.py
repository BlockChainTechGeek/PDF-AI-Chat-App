import streamlit as st
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
import joblib
import warnings
import traceback

# Set page config at the very beginning
st.set_page_config(page_title="AI PDF Chat", page_icon="📚", layout="wide")

from text_processing import (
    preprocess_text,
    split_text,
    truncate_text,
    select_relevant_chunks,
)
from audio_processing import record_audio, transcribe_audio
from embedding_utils import (
    load_embedding_model,
    create_or_load_index,
    save_index,
    add_chunks_to_index,
    add_conversation_to_index,
    similarity_search_cached,
)
from utils import (
    timed_execution,
    check_memory,
    get_conversation_context,
    get_cached_answer,
    set_cached_answer,
    get_pdf_text,
    load_qa_pipeline,
    load_llm,
    generate_answer_qa,
    generate_answer_llm,
)

warnings.filterwarnings("ignore", category=FutureWarning)

def process_query(question: str, context: str, method: str):
    if st.session_state.debug_mode:
        st.write(f"Debug: Processing query: {question}")
        st.write(f"Debug: Context length: {len(context)}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            truncated_context = truncate_text(context, max_tokens=500)

            if method == "Local LLM":
                answer = generate_answer_llm(st.session_state.llm, question, truncated_context)
            elif method == "QA Pipeline":
                answer = generate_answer_qa(st.session_state.qa_pipeline, question, truncated_context)
            elif method == "Both (Compare)":
                llm_answer = generate_answer_llm(st.session_state.llm, question, truncated_context)
                qa_answer = generate_answer_qa(st.session_state.qa_pipeline, question, truncated_context)
                answer = f"LLM Answer: {llm_answer}\n\nQA Pipeline Answer: {qa_answer}"
            else:
                raise ValueError(f"Invalid method: {method}")

            if not answer or len(answer.split()) < 5:
                raise ValueError("Generated answer is empty or too short")

            return answer

        except Exception as e:
            if st.session_state.debug_mode:
                st.write(f"Debug: Error in attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                st.error(f"Failed to generate an answer after {max_retries} attempts. Please try rephrasing your question.")
                return None

            # Adjust the question for retry
            question = f"Regarding the previous question '{question}', please provide a specific answer based on the given context."

    return None

def process_pdfs(pdf_docs: List[UploadedFile]):
    """Process uploaded PDF documents."""
    if pdf_docs:
        with st.spinner("Processing PDFs..."):
            try:
                raw_text = get_pdf_text(pdf_docs)
                st.write(f"Extracted {len(raw_text)} characters from PDFs")

                new_chunks = split_text(raw_text)
                st.write(f"Created {len(new_chunks)} text chunks")

                process_chunks(new_chunks)
                st.session_state.chunks = new_chunks  # Update session state
                st.success(f"✅ Processing complete! {len(new_chunks)} chunks added.")
            except Exception as e:
                st.error(f"❌ An error occurred while processing PDFs: {str(e)}")
                st.error(f"Error details: {traceback.format_exc()}")
    else:
        st.warning("⚠️ Please upload PDF documents before processing.")

def process_in_batches(chunks, batch_size=100):
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size]

def process_chunks(new_chunks: List[str]):
    """Process text chunks and add them to the index."""
    progress_bar = st.progress(0)
    for i, batch in enumerate(process_in_batches(new_chunks)):
        st.session_state.chunks.extend(batch)
        if "faiss_index" not in st.session_state or st.session_state.faiss_index is None:
            st.session_state.faiss_index = create_or_load_index(
                st.session_state.embedding_model.get_sentence_embedding_dimension()
            )
        st.session_state.faiss_index = add_chunks_to_index(
            st.session_state.faiss_index, batch, st.session_state.embedding_model
        )
        progress = (i + 1) / len(list(process_in_batches(new_chunks)))
        progress_bar.progress(progress)
    save_index(st.session_state.faiss_index)
    joblib.dump(st.session_state.chunks, "processed_chunks.joblib")

def reset_index():
    """Reset the index and conversation history."""
    if os.path.exists("faiss_index.npy"):
        os.remove("faiss_index.npy")
    if os.path.exists("processed_chunks.joblib"):
        os.remove("processed_chunks.joblib")
    st.session_state.faiss_index = create_or_load_index(
        st.session_state.embedding_model.get_sentence_embedding_dimension()
    )
    st.session_state.chunks = []
    st.session_state.conversation_history = []
    st.success("🔄 Index and conversation history reset successfully.")

def display_index_info():
    """Display information about the current index."""
    st.subheader("Index Information")
    if "faiss_index" in st.session_state and st.session_state.faiss_index is not None:
        st.write(f"📊 Vectors in index: {st.session_state.faiss_index.ntotal}")
    else:
        st.write("📊 No index available. Please process some documents.")
    
    if "chunks" in st.session_state:
        st.write(f"📄 Number of chunks: {len(st.session_state.chunks)}")
    else:
        st.write("📄 No chunks available. Please process some documents.")

def handle_user_input():
    """Handle user input for asking questions."""
    input_method = st.radio("Choose input method:", ("Text", "Voice"))

    if input_method == "Text":
        question = st.text_input("Ask a question about your documents:")
    else:
        question = handle_voice_input()

    if question:
        if "chunks" not in st.session_state or not st.session_state.chunks:
            st.warning("⚠️ Please process some documents before asking questions.")
            return

        with st.spinner("Thinking..."):
            try:
                conversation_context = get_conversation_context(st.session_state.conversation_history)
                full_query = f"{conversation_context}\nQ: {question}"
                relevant_chunks = select_relevant_chunks(full_query, st.session_state.chunks, top_k=7)
                context = " ".join(relevant_chunks)
                final_context = truncate_text(f"{conversation_context}\n\n{context}", max_tokens=1500)

                answer_method = st.session_state.get('answer_method', 'Local LLM')
                answer = process_query(question, final_context, answer_method)
                
                if answer:
                    st.write("Answer:", answer)
                    if "conversation_history" not in st.session_state:
                        st.session_state.conversation_history = []
                    st.session_state.conversation_history.append((question, answer))
                else:
                    st.error("Unable to generate an answer. Please try rephrasing your question.")
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")
                if st.session_state.debug_mode:
                    st.error(f"Error details: {traceback.format_exc()}")

def handle_voice_input():
    """Handle voice input for asking questions."""
    if st.button("Record Question"):
        audio_file = record_audio()
        question = transcribe_audio(audio_file)
        st.write(f"Transcribed question: {question}")
        os.remove(audio_file)
        return question
    return None

def display_conversation_history():
    """Display the recent conversation history."""
    st.subheader("Conversation History")
    if "conversation_history" in st.session_state and st.session_state.conversation_history:
        for i, (q, a) in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.expander(f"Q: {q}", expanded=i == 0):
                st.write("A:", a)
    else:
        st.write("No conversation history available.")

def main():
    # Initialize session state variables
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = load_embedding_model()
    if "llm" not in st.session_state:
        st.session_state.llm = load_llm()
    if "qa_pipeline" not in st.session_state:
        st.session_state.qa_pipeline = load_qa_pipeline()
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = create_or_load_index(
            st.session_state.embedding_model.get_sentence_embedding_dimension()
        )
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    st.title("AI PDF Chat Assistant")
    
    # Add logo
    logo_path = "Wallpaper.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type="pdf")

        if st.button("Process PDFs"):
            process_pdfs(pdf_docs)

        if st.button("Reset Index"):
            reset_index()

        display_index_info()

    with col2:
        st.subheader("Chat with your PDFs")
        handle_user_input()

    with col3:
        display_conversation_history()

    with st.sidebar:
        st.subheader("Settings")
        model_type = st.selectbox("Choose Model", ["Mistral-7B", "GPT-3.5", "GPT-4"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode")
        
        st.session_state.answer_method = st.radio(
            "Choose answer generation method:",
            ("Local LLM", "QA Pipeline", "Both (Compare)")
        )

if __name__ == "__main__":
    main()
