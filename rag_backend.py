import ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Initialize Ollama client
client = ollama.Client()

# Load and process your documents
loader = TextLoader("document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = Chroma.from_documents(docs, embeddings)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(query: Query):
    # Perform RAG
    relevant_docs = vectorstore.similarity_search(query.question, k=2)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""Context: {context}
    
    Question: {query.question}
    
    Answer:"""
    
    response = client.generate(model="llama3.2", prompt=prompt)
    return {"answer": response['response']}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
