"""
config.py - Configuration Management
=====================================
This file handles loading environment variables and storing configuration
constants used throughout the project.

WHY DO WE NEED THIS?
- API keys should NEVER be hardcoded in your code
- Environment variables keep secrets out of version control
- Centralizing config makes it easy to change settings in one place
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
# This reads the .env file and makes those values available via os.getenv()
load_dotenv()

# ─── API Key ─────────────────────────────────────────────────
# The Gemini API key is loaded from the .env file
# If it's not found, we raise a clear error message
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your-gemini-api-key-here":
    raise ValueError(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  ERROR: Google API Key not found!                          ║\n"
        "║                                                            ║\n"
        "║  Steps to fix:                                             ║\n"
        "║  1. Go to https://aistudio.google.com/app/apikey           ║\n"
        "║  2. Create a free API key                                  ║\n"
        "║  3. Copy .env.example to .env                              ║\n"
        "║  4. Paste your key in the .env file                        ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )

# ─── Model Settings ──────────────────────────────────────────
# Which Gemini model to use for generating answers
LLM_MODEL = "gemini-2.0-flash"

# Which Gemini model to use for creating embeddings (turning text into vectors)
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ─── Chunking Settings ───────────────────────────────────────
# When we load documents, we split them into smaller "chunks"
# These settings control how that splitting works

# Maximum number of characters in each chunk
CHUNK_SIZE = 500

# Number of characters that overlap between consecutive chunks
# Overlap ensures we don't lose context at chunk boundaries
CHUNK_OVERLAP = 100

# ─── Retrieval Settings ──────────────────────────────────────
# How many relevant chunks to retrieve when answering a question
TOP_K_RESULTS = 3

# ─── Database Settings ───────────────────────────────────────
# Directory where ChromaDB will store the vector database
CHROMA_DB_DIR = "chroma_db"

# Name of the collection (like a table) in ChromaDB
COLLECTION_NAME = "my_knowledge_base"
