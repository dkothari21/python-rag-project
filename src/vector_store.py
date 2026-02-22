"""
vector_store.py - Vector Database Management
==============================================
This file handles:
1. Creating embeddings from document chunks
2. Storing embeddings in ChromaDB (a local vector database)
3. Searching for relevant chunks based on a user's question

WHAT ARE EMBEDDINGS?
- Embeddings convert text into a list of numbers (a "vector")
- Similar texts produce similar vectors
- This lets us find relevant documents by comparing vectors mathematically

WHAT IS A VECTOR DATABASE?
- A specialized database for storing and querying vectors
- It uses algorithms to quickly find the most similar vectors
- ChromaDB stores everything locally on your computer (no cloud needed!)

HOW DOES SIMILARITY SEARCH WORK?
When you ask "What is a function in Python?":
1. Your question gets converted to a vector
2. ChromaDB compares it to all stored vectors
3. The chunks with the most similar vectors are returned
4. These become the "context" for the LLM to answer your question
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TOP_K_RESULTS,
)


def create_embedding_function():
    """
    Create the embedding function using Google Gemini.

    The embedding function converts text into vectors (lists of numbers).
    We use Google's embedding model which produces high-quality embeddings.

    Returns:
        A GoogleGenerativeAIEmbeddings object that can convert text to vectors
    """
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    return embedding_function


def create_vector_store(chunks: list) -> Chroma:
    """
    Create a vector store from document chunks.

    This function:
    1. Takes the text chunks from our documents
    2. Converts each chunk into a vector using the embedding function
    3. Stores all vectors in ChromaDB

    Args:
        chunks: List of Document chunks from split_documents()

    Returns:
        A Chroma vector store object (used for searching)

    Note:
        The first time you run this, it creates the database.
        Subsequent runs will add to the existing database.
    """
    print("\n🔄 Creating embeddings and storing in vector database...")
    print(f"   This may take a moment (processing {len(chunks)} chunks)...\n")

    # Get the embedding function
    embedding_function = create_embedding_function()

    # Create the vector store
    # ChromaDB will:
    #   - Convert each chunk to a vector using the embedding function
    #   - Store the vectors along with the original text
    #   - Save everything to disk in the CHROMA_DB_DIR folder
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"✅ Vector store created with {len(chunks)} chunks!")
    print(f"   Saved to: {CHROMA_DB_DIR}/")

    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing vector store from disk.

    Use this when you've already created the vector store and want to
    query it without re-processing all the documents.

    Returns:
        A Chroma vector store object (used for searching)
    """
    embedding_function = create_embedding_function()

    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def search_similar(vector_store: Chroma, query: str, top_k: int = TOP_K_RESULTS) -> list:
    """
    Search the vector store for chunks most similar to the query.

    This is the "Retrieval" part of RAG:
    1. Convert the query to a vector
    2. Find the top_k most similar chunks in the database
    3. Return those chunks (they become context for the LLM)

    Args:
        vector_store: The Chroma vector store to search
        query: The user's question
        top_k: Number of results to return (default from config)

    Returns:
        A list of (Document, similarity_score) tuples

    Example:
        results = search_similar(store, "What is a function?")
        for doc, score in results:
            print(f"Score: {score:.4f}")
            print(f"Content: {doc.page_content[:100]}...")
    """
    # similarity_search_with_score returns both the document and its similarity score
    # Lower score = more similar (it uses distance, not similarity)
    results = vector_store.similarity_search_with_score(query, k=top_k)

    return results
