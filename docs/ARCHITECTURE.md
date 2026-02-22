# 🏗️ Architecture Deep Dive

This document provides a deeper look at how the RAG system works, explaining each component and how data flows through the system.

---

## System Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │          python-rag-project          │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────────────┐
                    │                                      │
              ┌─────▼─────┐                        ┌──────▼──────┐
              │  INGESTION │                        │   QUERYING   │
              │  PIPELINE  │                        │   PIPELINE   │
              └─────┬─────┘                        └──────┬──────┘
                    │                                      │
         ┌──────────┼───────────┐           ┌──────────────┼──────────────┐
         │          │           │           │              │              │
    ┌────▼───┐ ┌───▼────┐ ┌───▼────┐  ┌───▼─────┐  ┌────▼────┐  ┌─────▼────┐
    │ Load   │ │ Split  │ │ Create │  │ Embed   │  │ Search  │  │ Generate │
    │ Docs   │ │ Chunks │ │ Embeds │  │ Question│  │ ChromaDB│  │ Answer   │
    │        │ │        │ │ & Store│  │         │  │         │  │ (Gemini) │
    └────────┘ └────────┘ └────────┘  └─────────┘  └─────────┘  └──────────┘
```

---

## Component Details

### 1. Configuration Layer (`src/config.py`)

```
Environment (.env)  ──▶  load_dotenv()  ──▶  Python Variables
                                              │
                                              ├── GOOGLE_API_KEY
                                              ├── LLM_MODEL
                                              ├── EMBEDDING_MODEL
                                              ├── CHUNK_SIZE
                                              ├── CHUNK_OVERLAP
                                              ├── TOP_K_RESULTS
                                              └── CHROMA_DB_DIR
```

**Why centralize config?**
- Change settings in **one place**, affects the entire app
- **Secrets stay in `.env`**, never in code
- Easy to switch models or tune parameters

---

### 2. Document Ingestion Pipeline

#### Step 1: Load Documents (`document_loader.py → load_documents()`)

```
data/sample_docs/
├── python_basics.txt  ──▶  TextLoader   ──▶  Document object
└── ai_ml_basics.txt   ──▶  TextLoader   ──▶  Document object
```

Each **Document** object contains:
- `page_content`: The actual text
- `metadata`: Information about the source (filename, path)

#### Step 2: Split into Chunks (`document_loader.py → split_documents()`)

```
Original Document (2000 chars)
│
├── Chunk 1: chars 0-500
├── Chunk 2: chars 400-900     ← 100 char overlap with Chunk 1
├── Chunk 3: chars 800-1300    ← 100 char overlap with Chunk 2
├── Chunk 4: chars 1200-1700   ← 100 char overlap with Chunk 3
└── Chunk 5: chars 1600-2000   ← 100 char overlap with Chunk 4
```

**Why overlap?** Consider this text split at position 500:

```
Chunk 1: "...Python uses try-except blocks for"
Chunk 2: "error handling. The syntax is..."
```

Without overlap, searching for "try-except error handling" might miss both chunks. With overlap, Chunk 2 would also contain "try-except blocks for error handling."

#### Step 3: Create Embeddings & Store (`vector_store.py → create_vector_store()`)

```
Text Chunk                          Embedding (Vector)
┌──────────────┐                   ┌────────────────────────┐
│"Python lists │   Gemini          │[0.12, -0.45, 0.78,    │
│ are ordered  │ ──Embedding──▶    │ 0.33, -0.21, 0.56,    │
│ collections" │   Model           │ ..., 0.89]             │
└──────────────┘                   └────────────────────────┘
                                            │
                                            ▼
                                     ┌──────────────┐
                                     │   ChromaDB    │
                                     │  (Local Disk) │
                                     └──────────────┘
```

---

### 3. Query Pipeline

#### Step 1: Embed the Question

```
"What is a list?"  ──▶  Gemini Embedding  ──▶  [0.15, -0.42, 0.81, ...]
```

#### Step 2: Similarity Search

```
Question Vector: [0.15, -0.42, 0.81, ...]

ChromaDB compares against all stored vectors:

Chunk 1 Vector: [0.12, -0.45, 0.78, ...]  ← Distance: 0.05 (VERY similar!) ✅
Chunk 2 Vector: [0.89, 0.23, -0.61, ...]  ← Distance: 0.82 (not similar)
Chunk 3 Vector: [0.11, -0.40, 0.75, ...]  ← Distance: 0.08 (similar!) ✅
Chunk 4 Vector: [-0.56, 0.71, 0.12, ...]  ← Distance: 0.91 (not similar)
Chunk 5 Vector: [0.14, -0.38, 0.80, ...]  ← Distance: 0.06 (similar!) ✅

Top 3 results returned (lowest distance = most similar)
```

#### Step 3: Build Prompt with Context

```
┌────────────────────────────────────────────────────┐
│                 PROMPT TO GEMINI                    │
│                                                    │
│  System: You are a helpful assistant that answers  │
│  questions based on the provided context.          │
│                                                    │
│  CONTEXT:                                          │
│  [Chunk 1 text]                                    │
│  [Chunk 3 text]                                    │
│  [Chunk 5 text]                                    │
│                                                    │
│  QUESTION: What is a list in Python?               │
│                                                    │
│  ANSWER:                                           │
└────────────────────────────────────────────────────┘
```

#### Step 4: Gemini Generates Answer

Gemini reads the context chunks and the question, then generates an answer that is **grounded in your documents** rather than making things up.

---

## Data Flow Summary

```
┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Documents│──▶│ Text       │──▶│ Embedding│──▶│ ChromaDB │──▶│ Retriever│
│ (.txt,   │   │ Chunks     │   │ Vectors  │   │ Storage  │   │ Search   │
│  .pdf)   │   │ (500 chars)│   │ (numbers)│   │ (local)  │   │          │
└──────────┘   └────────────┘   └──────────┘   └──────────┘   └─────┬────┘
                                                                      │
                                                                      ▼
┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Final    │◀──│ Gemini LLM │◀──│ Prompt + │◀──│ Top K    │◀──│ Question │
│ Answer   │   │ Generation │   │ Context  │   │ Chunks   │   │ Embedding│
└──────────┘   └────────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## Technology Choices & Why

| Technology | Purpose | Why This Choice? |
|-----------|---------|------------------|
| **Python** | Language | Most popular for AI/ML, rich ecosystem |
| **Google Gemini** | LLM + Embeddings | Free tier, powerful, single API for both |
| **ChromaDB** | Vector Database | Local (no account), easy setup, beginner-friendly |
| **LangChain** | Framework | Simplifies RAG pipeline, huge community |
| **python-dotenv** | Config | Industry standard for managing secrets |

---

## Configuration Tuning Guide

| Setting | Default | Effect of Increasing | Effect of Decreasing |
|---------|---------|---------------------|---------------------|
| `CHUNK_SIZE` | 500 | Broader context per chunk, less precise | More precise chunks, might lose context |
| `CHUNK_OVERLAP` | 100 | Better continuity, slightly more storage | Might miss info at boundaries |
| `TOP_K_RESULTS` | 3 | More context for LLM, slower, may add noise | Faster, more focused, might miss info |
| `temperature` | 0.3 | More creative/varied answers | More focused/deterministic answers |

---

## Security Notes

1. **API Key** — Never commit your `.env` file. The `.gitignore` handles this.
2. **ChromaDB** — Data is stored locally in `chroma_db/`. Don't commit this to GitHub if it contains sensitive data.
3. **Documents** — Be mindful of what documents you add. They are processed and stored locally.
