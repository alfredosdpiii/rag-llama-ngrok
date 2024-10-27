# RAG System with Ollama

A Retrieval-Augmented Generation (RAG) system that processes local documents and answers questions using Ollama's local LLM. The system supports multiple document formats, maintains a persistent vector database, and provides both CLI and API interfaces.

## Features

- **Multi-format Document Support**:

  - PDF (.pdf)
  - Text (.txt)
  - CSV (.csv)
  - JSON (.json)
  - Markdown (.md)

- **Vector Database**:

  - Persistent storage
  - Automatic loading of previous sessions
  - Document chunk management
  - Configurable retrieval size

- **User Interfaces**:
  - Interactive CLI with rich formatting
  - RESTful API with FastAPI
  - Public access via ngrok tunneling

## Prerequisites

### System Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update && sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic1
```

#### macOS

```bash
brew install \
    tesseract \
    poppler \
    magic
```

#### Windows

1. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install [Poppler](http://blog.alivate.com.au/poppler-windows/)
3. Add both to your system PATH

### Required Software

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- `llama2` model pulled in Ollama (`ollama pull llama2`)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd rag-system
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

## Usage

### Running the System

```bash
python rag_backend.py
```

### CLI Menu Options

1. **Process Documents**:

   - Select directory containing documents
   - View processing status
   - See processing results

2. **Run Server**:

   - Start FastAPI server
   - Get public URL via ngrok
   - Access API endpoints

3. **View Vector Store Info**:

   - See database location
   - View document count
   - Check storage size

4. **Exit**

### API Endpoints

#### Query Documents

```bash
curl -X POST "https://your-ngrok-url/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here", "k": 3}'
```

or through localhost

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here", "k": 3}'
```

- `question`: The query to answer
- `k`: Number of relevant chunks to retrieve (default: 3)

#### Get Supported Formats

```bash
curl "https://your-ngrok-url/supported-formats"
```

#### Get Vector Store Info

```bash
curl "https://your-ngrok-url/vector-store-info"
```

## Directory Structure

```
.
├── rag_backend.py     # Main application file
├── requirements.txt   # Python dependencies
├── Documents/         # Default documents directory
└── chroma_db/        # Vector database storage
```

## Configuration

The vector database is stored in `./chroma_db` by default. You can:

- Backup this directory to preserve your processed documents
- Delete it to start fresh
- Monitor its size through the UI

## Troubleshooting

1. **PDF Processing Issues**:

   - Ensure Tesseract and Poppler are properly installed
   - Check file permissions
   - Verify PDF is not password-protected

2. **Ollama Connection**:

   - Verify Ollama is running: `curl http://localhost:11434/api/tags`
   - Check if llama2 model is pulled: `ollama list`

3. **Vector Store Issues**:
   - Check disk space
   - Verify write permissions in the chroma_db directory

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[MIT License](LICENSE)
