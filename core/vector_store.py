import os
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"

# Use the env variable, but fallback to the default Gemini model if it's not set.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def _reset_collection():
    """Delete persisted collection so dimension changes take effect."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

def build_vector_store(transcript: str) -> Chroma:
    print(f"Building vector store with API embeddings: {EMBEDDING_MODEL} ...")
    _reset_collection()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(splitter.split_text(transcript))
    ]
    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

def load_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )

def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )