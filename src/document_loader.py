"""
document_loader.py - Loading & Chunking Documents
===================================================
This file handles:
1. Reading documents from a folder (supports .txt and .pdf files)
2. Splitting documents into smaller chunks for better retrieval

WHY DO WE SPLIT DOCUMENTS?
- LLMs have a limited context window (max tokens they can process)
- Smaller chunks = more precise retrieval (find exactly the relevant part)
- Overlapping chunks ensure we don't lose information at boundaries

WHAT IS A "CHUNK"?
Think of it like cutting a book into index cards. Each card has a portion
of the text. When someone asks a question, we find the most relevant cards
instead of searching the entire book.
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(directory_path: str) -> list:
    """
    Load all .txt and .pdf documents from a directory.

    Args:
        directory_path: Path to the folder containing documents

    Returns:
        A list of Document objects (each contains the text + metadata)

    Example:
        documents = load_documents("data/sample_docs")
        print(f"Loaded {len(documents)} documents")
    """
    documents = []

    # Check if the directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"Directory '{directory_path}' not found! "
            f"Make sure you have documents in this folder."
        )

    # Walk through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Skip directories and hidden files
        if os.path.isdir(file_path) or filename.startswith("."):
            continue

        try:
            if filename.endswith(".txt"):
                # TextLoader reads plain text files
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
                print(f"  ✅ Loaded: {filename}")

            elif filename.endswith(".pdf"):
                # PyPDFLoader reads PDF files (one document per page)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"  ✅ Loaded: {filename}")

            else:
                print(f"  ⏭️  Skipped (unsupported format): {filename}")

        except Exception as e:
            print(f"  ❌ Error loading {filename}: {e}")

    if not documents:
        print("⚠️  No documents were loaded! Add .txt or .pdf files to the directory.")

    return documents


def split_documents(documents: list) -> list:
    """
    Split documents into smaller, overlapping chunks.

    This uses RecursiveCharacterTextSplitter, which tries to split on:
    1. Paragraphs (double newlines)
    2. Sentences (periods)
    3. Words (spaces)
    4. Characters (as a last resort)

    This hierarchy ensures chunks are as meaningful as possible.

    Args:
        documents: List of Document objects from load_documents()

    Returns:
        A list of smaller Document chunks

    Example:
        docs = load_documents("data/sample_docs")
        chunks = split_documents(docs)
        print(f"Split into {len(chunks)} chunks")
    """
    # Create the text splitter with our configured settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,         # Max characters per chunk
        chunk_overlap=CHUNK_OVERLAP,   # Characters of overlap between chunks
        length_function=len,           # How to measure chunk length
        add_start_index=True,          # Track where each chunk starts in the original doc
    )

    # Split all documents into chunks
    chunks = text_splitter.split_documents(documents)

    print(f"\n📄 Split {len(documents)} documents into {len(chunks)} chunks")
    print(f"   (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return chunks
