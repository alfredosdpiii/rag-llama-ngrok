import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok
import logging
import os
import sys
from typing import List, Dict
from pathlib import Path
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

# Set up logging with rich console
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_default_documents_path() -> str:
    """Get the default documents directory for the current operating system."""
    return os.path.join(os.path.expanduser("~"), "Documents")


def normalize_path(path: str) -> Path:
    """Normalize a path string to a proper Path object."""
    expanded_path = os.path.expanduser(path)
    expanded_path = os.path.expandvars(expanded_path)
    abs_path = os.path.abspath(expanded_path)
    return Path(abs_path)


class DocumentProcessor:
    SUPPORTED_EXTENSIONS = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".json": JSONLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    def __init__(self, docs_dir: str = "Documents", persist_directory: str = None):
        self.docs_dir = normalize_path(docs_dir)
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.vectorstore = None
        # Set default persist_directory if none provided
        self.persist_directory = persist_directory or os.path.join(
            os.getcwd(), "chroma_db"
        )

    def get_loader_for_file(self, file_path: Path):
        """Get appropriate loader for file type with enhanced PDF handling."""
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")

        loader_class = self.SUPPORTED_EXTENSIONS[extension]

        if extension == ".pdf":
            return PyPDFLoader(str(file_path))
        elif extension == ".json":
            return loader_class(str(file_path), jq_schema=".", text_content=False)

        return loader_class(str(file_path))

    def process_documents(self) -> Dict[str, List[str]]:
        """Process all supported documents in the documents directory."""
        all_docs = []
        processed_files = []
        failed_files = []

        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Get list of files to process
        files_to_process = [
            f
            for f in self.docs_dir.glob("*")
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not files_to_process:
            console.print(
                "No supported documents found in the directory.", style="yellow"
            )
            return {"processed": [], "failed": []}

        # Process files with progress display
        for file_path in files_to_process:
            console.print(f"\nProcessing: {file_path.name}")
            try:
                with Status("Loading document...", console=console):
                    loader = self.get_loader_for_file(file_path)
                    documents = loader.load()
                    if not documents:
                        raise ValueError("No text content extracted from document")
                    console.print(
                        f"Extracted {len(documents)} pages/sections", style="blue"
                    )

                with Status("Splitting text...", console=console):
                    split_docs = self.text_splitter.split_documents(documents)
                    if not split_docs:
                        raise ValueError("Document splitting produced no results")
                    all_docs.extend(split_docs)
                    processed_files.append(str(file_path))
                    console.print(
                        f"Created {len(split_docs)} text chunks", style="blue"
                    )

                console.print(
                    f"✅ Successfully processed {file_path.name}", style="green"
                )

            except Exception as e:
                console.print(
                    f"❌ Error processing {file_path.name}: {str(e)}", style="red"
                )
                failed_files.append(str(file_path))
                continue

        if not all_docs:
            raise ValueError("No documents were successfully processed")

        # Create vector store
        with Status("Creating vector store...", console=console):
            console.print(
                f"Creating vector store with {len(all_docs)} documents...", style="blue"
            )
            console.print(
                f"Vector store location: {self.persist_directory}", style="blue"
            )

            self.vectorstore = Chroma.from_documents(
                documents=all_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )

        console.print("✅ Vector store successfully created!", style="green")

        return {"processed": processed_files, "failed": failed_files}

    def load_existing_vectorstore(self) -> bool:
        """Try to load an existing vector store."""
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
                return True
            return False
        except Exception as e:
            console.print(f"Error loading existing vector store: {e}", style="red")
            return False

    def query_documents(self, question: str, k: int = 2) -> Dict:
        """Query the vector store with a question."""
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialized. Please process documents first."
            )

        relevant_docs = self.vectorstore.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        return {
            "context": context,
            "docs": [doc.metadata for doc in relevant_docs],
            "num_chunks": k,
        }


class Query(BaseModel):
    question: str
    k: int = 3  # Default to 3 chunks for better context


class ProcessingStatus(BaseModel):
    status: str
    processed_files: List[str]


class RAGSystem:
    def __init__(self):
        self.persist_directory = os.path.join(os.getcwd(), "chroma_db")
        self.doc_processor = DocumentProcessor(persist_directory=self.persist_directory)
        self.processed = self.doc_processor.load_existing_vectorstore()


def create_app(doc_processor: DocumentProcessor) -> FastAPI:
    app = FastAPI(title="RAG API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/query")
    async def query(query: Query):
        """
        Query the processed documents.
        - question: The question to ask
        - k: Number of relevant chunks to retrieve (default: 3)
        """
        try:
            if not doc_processor.vectorstore:
                raise HTTPException(
                    status_code=400, detail="No documents processed yet"
                )

            # Validate k
            if query.k < 1:
                raise HTTPException(status_code=400, detail="k must be at least 1")
            if query.k > 10:
                raise HTTPException(
                    status_code=400, detail="k cannot be greater than 10"
                )

            result = doc_processor.query_documents(query.question, k=query.k)

            client = ollama.Client(host="http://localhost:11434")
            prompt = f"""Context: {result['context']}

            Question: {query.question}

            Answer:"""

            response = client.generate(model="llama3.2", prompt=prompt)

            return {
                "answer": response["response"],
                "sources": result["docs"],
                "chunks_used": query.k,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/supported-formats")
    async def get_supported_formats():
        """Return list of supported file formats."""
        return {"formats": list(DocumentProcessor.SUPPORTED_EXTENSIONS.keys())}

    @app.get("/vector-store-info")
    async def get_vector_store_info():
        """Get information about the vector store."""
        try:
            collection = doc_processor.vectorstore._collection
            return {
                "location": doc_processor.persist_directory,
                "document_count": collection.count() if collection else 0,
                "exists": os.path.exists(doc_processor.persist_directory),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def process_documents_cli(rag_system: RAGSystem):
    """CLI interface for processing documents."""
    default_path = get_default_documents_path()

    console.print(f"\nCurrent working directory: {os.getcwd()}", style="blue")
    console.print(f"Default documents directory: {default_path}", style="blue")

    docs_path = questionary.path(
        "Enter the path to your documents directory:",
        only_directories=True,
        default=default_path,
    ).ask()

    if not docs_path:
        return

    docs_path = normalize_path(docs_path)

    if not docs_path.exists():
        console.print(f"\nDirectory not found: {docs_path}", style="yellow")
        if questionary.confirm("Would you like to create this directory?").ask():
            try:
                docs_path.mkdir(parents=True, exist_ok=True)
                console.print(f"Created directory: {docs_path}", style="green")
            except Exception as e:
                console.print(f"Failed to create directory: {e}", style="red")
                return
        else:
            return

    try:
        rag_system.doc_processor = DocumentProcessor(
            str(docs_path), persist_directory=rag_system.persist_directory
        )
        console.print(f"\nProcessing documents in: {docs_path}", style="blue")
        console.print(
            f"Vector store will be saved in: {rag_system.persist_directory}",
            style="blue",
        )

        supported_files = [
            f
            for f in docs_path.glob("*")
            if f.suffix.lower() in DocumentProcessor.SUPPORTED_EXTENSIONS
        ]

        if not supported_files:
            console.print(
                f"\nNo supported documents found in: {docs_path}", style="yellow"
            )
            console.print("\nSupported formats:", style="blue")
            for ext in DocumentProcessor.SUPPORTED_EXTENSIONS.keys():
                console.print(f"  - {ext}")
            return

        console.print("\nFound files:", style="green")
        for file in supported_files:
            console.print(f"  - {file.name}")

        if questionary.confirm("Proceed with processing these files?").ask():
            result = rag_system.doc_processor.process_documents()
            rag_system.processed = True

            console.print("\n📊 Processing Summary:", style="bold blue")
            if result["processed"]:
                console.print("\n✅ Successfully processed files:", style="green")
                for file in result["processed"]:
                    console.print(f"  - {Path(file).name}")

            if result["failed"]:
                console.print("\n❌ Failed to process files:", style="red")
                for file in result["failed"]:
                    console.print(f"  - {Path(file).name}")
        else:
            console.print("\nDocument processing cancelled.", style="yellow")

    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}", style="red")


def run_server_cli(rag_system: RAGSystem):
    """CLI interface for running the server."""
    if not rag_system.processed:
        console.print("\n⚠️ No documents have been processed yet!", style="yellow")
        if not questionary.confirm(
            "Do you want to continue without processed documents?"
        ).ask():
            return

    try:
        app = create_app(rag_system.doc_processor)

        # Try to set up ngrok tunnel
        ngrok_tunnel = None
        try:
            ngrok_tunnel = ngrok.connect(8000)
            console.print(f"\n🌐 Public URL: {ngrok_tunnel.public_url}", style="green")
        except Exception as e:
            console.print(f"\n⚠️  Error setting up ngrok: {str(e)}", style="yellow")
            console.print("Continuing with local server only...", style="yellow")

        # Run the server
        console.print("\n🚀 Starting server...", style="green")
        uvicorn.run(app, host="0.0.0.0", port=8000)

    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}", style="red")
        if ngrok_tunnel:
            try:
                ngrok_tunnel.close()
            except:
                pass


def show_vectorstore_info(rag_system: RAGSystem):
    """Display information about the vector store."""
    try:
        console.print("\n📊 Vector Store Information:", style="bold blue")
        console.print(f"Location: {rag_system.persist_directory}", style="blue")

        if rag_system.processed:
            try:
                collection = rag_system.doc_processor.vectorstore._collection
                count = collection.count()
                console.print(f"Number of documents: {count}", style="green")
                size = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, _, files in os.walk(rag_system.persist_directory)
                    for file in files
                )
                console.print(f"Total size: {size / 1024 / 1024:.2f} MB", style="green")
            except Exception as e:
                console.print(f"Error getting vector store stats: {e}", style="red")
        else:
            console.print("No vector store has been created yet.", style="yellow")

        try:
            questionary.press_any_key_to_continue(
                message="\nPress any key to continue..."
            ).ask()
        except EOFError:
            # Handle the case where standard input is closed
            console.print("\nReturning to menu...", style="yellow")
            return

    except Exception as e:
        console.print(f"\n❌ Error displaying vector store info: {str(e)}", style="red")
        console.print("\nReturning to menu...", style="yellow")


def run_server_cli(rag_system: RAGSystem):
    """CLI interface for running the server."""
    if not rag_system.processed:
        console.print("\n⚠️ No documents have been processed yet!", style="yellow")
        if not questionary.confirm(
            "Do you want to continue without processed documents?"
        ).ask():
            return

    try:
        app = create_app(rag_system.doc_processor)

        # Try to set up ngrok tunnel
        ngrok_tunnel = None
        try:
            ngrok_tunnel = ngrok.connect(8000)
            console.print(f"\n🌐 Public URL: {ngrok_tunnel.public_url}", style="green")
        except Exception as e:
            console.print(f"\n⚠️  Error setting up ngrok: {str(e)}", style="yellow")
            console.print("Continuing with local server only...", style="yellow")

        # Run the server
        console.print("\n🚀 Starting server...", style="green")
        try:
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except KeyboardInterrupt:
            console.print("\n⚠️ Server shutdown requested", style="yellow")
        finally:
            if ngrok_tunnel:
                try:
                    ngrok_tunnel.close()
                except:
                    pass
            # Reset terminal state
            console.input_enabled = True

    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}", style="red")
        if ngrok_tunnel:
            try:
                ngrok_tunnel.close()
            except:
                pass


def main_menu():
    """Main CLI menu."""
    console.print(Panel.fit("🤖 RAG System CLI", style="bold blue"))

    rag_system = RAGSystem()

    # Show vector store location at startup
    console.print(
        f"\nVector store location: {rag_system.persist_directory}", style="blue"
    )
    if rag_system.processed:
        console.print("✅ Loaded existing vector store", style="green")

    while True:
        try:
            status = "🟢" if rag_system.processed else "🔴"
            choice = questionary.select(
                "What would you like to do?",
                choices=[
                    f"1. Process Documents (Documents Status: {status})",
                    "2. Run Server",
                    "3. View Vector Store Info",
                    "4. Exit",
                ],
            ).ask()

            if choice is None:  # Handle Ctrl+C in menu
                raise KeyboardInterrupt

            if choice.startswith("1."):
                process_documents_cli(rag_system)
            elif choice == "2. Run Server":
                run_server_cli(rag_system)
            elif choice == "3. View Vector Store Info":
                show_vectorstore_info(rag_system)
            elif choice == "4. Exit":
                console.print("\n👋 Goodbye!", style="green")
                sys.exit(0)

        except EOFError:
            console.print("\n⚠️ Input error, returning to menu...", style="yellow")
            continue
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="green")
            sys.exit(0)


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style="green")
        sys.exit(0)
