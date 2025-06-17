# RAG Document Processing System

A Retrieval-Augmented Generation (RAG) system built with Chainlit that allows you to upload documents (PDF, DOCX, PPTX), process them into chunks, store embeddings in ChromaDB, and query them using a local LLM.

## Features

- **Document Upload**: Support for PDF, DOCX, and PPTX files
- **Text Processing**: Automatic text extraction and chunking (~200 words per chunk)
- **Vector Storage**: ChromaDB for storing document embeddings
- **Local LLM**: Uses llama.cpp with Gemma-3-4B-IT model
- **Embeddings**: SentenceTransformers (Lajavaness/sentence-camembert-large model) for text embeddings
- **Interactive UI**: Chainlit-based web interface
- **File Management**: Upload files via deposit zone or chat attachments

## Prerequisites

- Python 3.8+
- llama.cpp installed and available in PATH
- Virtual environment (recommended)

## Installation

1. **Clone/Download the project**
   ```bash
   cd /path/to/your/project
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Models Setup

The system requires two models to be available through llama.cpp:

1. **LLM Model**: `ggml-org/gemma-3-4b-it-GGUF`
2. **Embedding Model**: SentenceTransformers will automatically download the `Lajavaness/sentence-camembert-large` model on first use

These will be automatically downloaded by llama-server when first used.

## Usage

### Quick Start (Recommended)

Use the provided script to start all services at once:

```bash
./run_services.sh
```

This will start:
- LLM server on port 8080
- Embedding server on port 8081
- Chainlit web interface

### Manual Start

If you prefer to start services manually:

1. **Start the LLM server**
   ```bash
   source venv/bin/activate
   llama-server -hf ggml-org/gemma-3-4b-it-GGUF -c 4096
   ```

2. **Start the embedding server** (in a new terminal)
   ```bash
   source venv/bin/activate
   llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --embeddings --port 8081
   ```

3. **Start the Chainlit application** (in a new terminal)
   ```bash
   source venv/bin/activate
   chainlit run main.py
   ```

### Using the Application

1. **Access the web interface** at `http://localhost:8000`

2. **Upload documents**:
   - On first visit, use the file deposit zone to upload initial documents
   - During chat, attach files using the attachment button in the chat bar

3. **Query your documents**:
   - Type questions about your uploaded documents
   - The system will find relevant chunks and generate answers
   - Referenced documents will be attached to responses

## Project Structure

```
├── main.py                 # Chainlit application entry point
├── requirements.txt        # Python dependencies
├── run_services.sh        # Script to start all services
├── chainlit.md           # Chainlit configuration
├── utils/
│   ├── chromadb_storage.py    # ChromaDB vector storage utilities
│   ├── doc_parser.py          # Document parsing and chunking
│   ├── llama_cpp_call.py      # LLM and embedding API calls
│   └── type_parser.py         # File type detection
└── chroma_db/            # ChromaDB persistence directory
```

## Configuration

### Server Endpoints

- **LLM Server**: `http://127.0.0.1:8080/v1`
- **Embedding Server**: `http://127.0.0.1:8081/v1`

These can be modified in `utils/llama_cpp_call.py` if needed.

### File Upload Limits

- **Max file size**: 500MB
- **Max files per upload**: 3
- **Supported formats**: PDF, DOCX, PPTX

These can be adjusted in `main.py` and `.chainlit/config.toml`.

### Chunking Settings

- **Default chunk size**: 200 words
- **Overlap**: None (configurable in `utils/doc_parser.py`)

## Troubleshooting

### Common Issues

1. **ChromaDB not persisting data**
   - Ensure the `chroma_db` directory has write permissions
   - Check if files are being processed successfully

2. **LLM/Embedding server connection errors**
   - Verify servers are running on correct ports (8080, 8081)
   - Check if models are downloaded correctly
   - Increase timeout in `llama_cpp_call.py` if models are slow

3. **File upload/parsing errors**
   - Ensure all required packages are installed (PyPDF2, python-docx, python-pptx)
   - Check file format compatibility
   - Verify file paths are accessible

### Logs and Debugging

- Check terminal output for server logs
- Chainlit logs appear in the terminal where `chainlit run` was executed
- Enable debug mode by setting environment variables if needed

## Dependencies

See `requirements.txt` for the complete list. Key dependencies include:

- `chainlit`: Web interface framework
- `chromadb`: Vector database
- `httpx`: Async HTTP client
- `PyPDF2`: PDF processing
- `python-docx`: DOCX processing
- `python-pptx`: PPTX processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.
