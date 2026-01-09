#Vector embedding
from langchain_ollama import OllamaEmbeddings   # Embedding model (text to vector)
from langchain_chroma import Chroma             # Vector store 
from langchain_core.documents import Document   # Create documents and add to chroma database
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#1. Chunking/indexing
# Load book
loader = PyPDFLoader("Age of Revolution.pdf")
data = loader.load()  

# Prepare embeddings model
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

# Vector data base location
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Add data to docs if location exists
if add_documents:
    documents = []
    ids = []
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap = 150)
    chunks = splitter.split_documents(data)
    
    for i, chunk in enumerate(chunks):
        # Each chunk is a Langchain doc with page text + metadata
        document = Document(
            page_content = chunk.page_content,
            metadata= {
                "source": chunk.metadata.get(
                    "source", "Age of Revolution.pdf"),
                "page": chunk.metadata.get("page", "NA")
            }
        )
        documents.append(document)
        ids.append(str(i))
    
    # Create vector (Indexing)
    db = Chroma.from_documents(documents, embeddings,
                               persist_directory = db_location)
    print(f"Added {len(documents)} chunks to new Chroma DB at \
    {db_location}")

else:
    # If db already exists, load it
    db = Chroma(persist_directory = db_location,
                embedding_function =  embeddings)
    print(f"Loaded existing Chroma DB from {db_location}")

# 2 Retrieving
retriever = db.as_retriever(
    search_kwargs = {"k": 7})
