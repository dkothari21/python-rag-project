"""
main.py - Entry Point for the RAG Application
================================================
This is the main file that runs the entire RAG pipeline.

It does two things:
1. INGEST MODE: Load documents, split them, and store in the vector database
2. QUERY MODE: Ask questions and get answers from the RAG system

Run this file to start the application:
    python main.py
"""

import sys
import os

# Add the project root to the Python path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import load_documents, split_documents
from src.vector_store import create_vector_store
from src.rag_chain import create_rag_chain, ask_question


def ingest_documents():
    """
    Step 1: Load and process documents into the vector database.

    This is the "preparation" step that you run once (or whenever
    you add new documents). It:
    1. Reads all documents from data/sample_docs/
    2. Splits them into smaller chunks
    3. Creates embeddings and stores them in ChromaDB
    """
    print("=" * 60)
    print("📥 DOCUMENT INGESTION")
    print("=" * 60)
    print("\nLoading documents from data/sample_docs/...\n")

    # Load all documents from the sample_docs directory
    documents = load_documents("data/sample_docs")

    if not documents:
        print("No documents found! Add .txt or .pdf files to data/sample_docs/")
        return

    # Split documents into chunks
    chunks = split_documents(documents)

    # Create the vector store (this also creates the embeddings)
    create_vector_store(chunks)

    print("\n✅ Ingestion complete! You can now ask questions.\n")


def interactive_query():
    """
    Step 2: Interactive question-answering loop.

    This creates the RAG chain and lets you ask questions in a loop.
    Type 'quit' or 'exit' to stop.
    """
    print("=" * 60)
    print("🤖 RAG QUESTION-ANSWERING")
    print("=" * 60)

    # Create the RAG chain (loads the vector store + connects to Gemini)
    chain = create_rag_chain()

    print("Ask me anything about the documents in the knowledge base!")
    print("Type 'quit' or 'exit' to stop.\n")
    print("-" * 60)

    while True:
        # Get question from user
        question = input("\n❓ Your question: ").strip()

        # Check for exit commands
        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        # Skip empty questions
        if not question:
            print("Please enter a question.")
            continue

        print("\n🔍 Searching knowledge base and generating answer...\n")

        try:
            # Ask the question and get the response
            response = ask_question(chain, question)

            # Display the answer
            print("─" * 60)
            print("💡 ANSWER:")
            print("─" * 60)
            print(response["result"])

            # Display the source documents used
            print("\n📄 SOURCES USED:")
            print("─" * 60)
            for i, doc in enumerate(response["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                print(f"\n  Source {i}: {source}")
                # Show first 150 characters of the chunk
                preview = doc.page_content[:150].replace("\n", " ")
                print(f"  Preview: {preview}...")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure you've run ingestion first and your API key is valid.")

        print("\n" + "-" * 60)


def main():
    """
    Main entry point - runs the full pipeline.
    """
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         🧠 Python RAG with Google Gemini                ║")
    print("║         Retrieval-Augmented Generation Demo             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Check if vector store already exists
    if os.path.exists("chroma_db"):
        print("📦 Existing vector database found!")
        choice = input("Do you want to re-ingest documents? (y/n): ").strip().lower()
        if choice == "y":
            ingest_documents()
    else:
        # First run - must ingest documents
        print("📦 No vector database found. Starting document ingestion...\n")
        ingest_documents()

    # Start the interactive Q&A
    interactive_query()


if __name__ == "__main__":
    main()
