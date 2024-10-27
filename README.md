# rag-llama-ngrok

# Requirements

- `pip install requirements.txt`

System-level dependencies you'll also need:
For Ubuntu/Debian:

```bashCopysudo apt-get update && sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic1 \
    libreoffice \
    pandoc
```

For macOS:

```bashCopybrew install \
    tesseract \
    poppler \
    magic \
    libreoffice \
    pandoc
```

For Windows:

Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
Install Poppler from: http://blog.alivate.com.au/poppler-windows/
Install LibreOffice from: https://www.libreoffice.org/download/download/
Install Pandoc from: https://pandoc.org/installing.html

```bashCopysudo
curl -X POST "https://your-ngrok-url/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Elixir?", "k": 3}'
```
