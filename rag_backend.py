import ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Ollama client
client = ollama.Client(host="http://localhost:11434")

# Load and process your documents
loader = TextLoader("document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3.2")  # Change to the correct model name
vectorstore = Chroma.from_documents(docs, embeddings)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(query: Query):
    try:
        # Perform RAG
        relevant_docs = vectorstore.similarity_search(query.question, k=2)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""Context: {context}
        
        Question: {query.question}
        
        Answer:"""
        
        response = client.generate(model="llama3.2", prompt=prompt)  # Change to the correct model name
        return {"answer": response['response']}
    except Exception as e:
        logging.error(f"Error in query processing: {str(e)}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    try:
        # Set up ngrok
        ngrok_tunnel = ngrok.connect(8000)
        logging.info(f"Public URL: {ngrok_tunnel.public_url}")
    except Exception as e:
        logging.error(f"Error setting up ngrok: {str(e)}")
        ngrok_tunnel = None
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
