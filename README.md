# AI PDF Chat Assistant

This project is an AI-powered PDF Chatbot Assistant that handles complex operations like multiple PDF document uploads, real-time audio recording (using OpenAI's Whisper model), and dynamic chat interactions. You can ask questions about the PDFs using natural language, and the chatbot can provide comprehensive summaries of entire documents in a single interaction, streamlining information retrieval.

## Features

- Multiple PDF document upload and processing
- Text and voice-based question input using OpenAI's Whisper model
- AI-powered question answering using local LLM and QA pipeline
- Conversation history tracking and display
- Configurable settings:
  - Adjustable GPT model temperature for optimal performance
  - User-selectable answer generation methods (Local LLM, QA Pipeline, or both)
- Built with Streamlit for a user-friendly interface

## Required Models

This project uses the following open-source models by default:

- Mistral 7B: A powerful Large Language Model for text generation with 7 billion parameters.
  Download from: [TheBloke Mistral 7B on Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF)
- Phi-2: A Transformer-based Model developed by Microsoft Research with 2.7 billion parameters.
  Download from: [TheBloke Phi-2 on Hugging Face](https://huggingface.co/TheBloke/phi-2-GGUF)

## Using Alternative Models

You can also use your preferred language model if you believe it will be more effective for your use case. To use a custom model:

- Ensure your model is compatible with the llama.cpp format (.gguf file).
- Download your chosen model and place it in a directory of your choice.
- Update the MODEL_PATH in the .env file to point to your custom model.

### Example:

- MODEL_PATH=/path/to/your/custom_model.gguf
- Note: The performance and compatibility of custom models may vary. Ensure your system meets the requirements for running your chosen model.

## Installation

1. Clone this repository:
   git clone https://github.com/yourusername/ai-pdf-chat-assistant.git
   cd ai-pdf-chat-assistant
   

2. Install required packages:
   pip install -r requirements.txt

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Edit `.env` and set the correct paths and values

4. Download the LLM model:
   - Download the Mistral 7B model (or your preferred model)
   - Place it in a directory and update the MODEL_PATH in your .env file

5. Run the Streamlit app:
   streamlit run app.py

## Usage

1. Launch the app using the command above
2. Upload PDF documents using the file uploader in the sidebar
3. Click "Process PDFs" to extract and index the content
4. Ask questions about the documents using text input or voice recording
5. View AI-generated answers and the conversation history in the main panel
6. Adjust settings in the sidebar as needed for optimal performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
