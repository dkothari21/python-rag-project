# 🧠 Python RAG with Google Gemini

A beginner-friendly Retrieval-Augmented Generation (RAG) application built with Python, Google Gemini AI, and ChromaDB.

> **What is RAG?** RAG stands for **Retrieval-Augmented Generation**. It is a technique that makes AI smarter by giving it access to your own documents. Instead of relying only on what the AI was trained on, RAG first searches your documents for relevant information, then uses that information to generate a more accurate answer.

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [How RAG Works (Visual Explanation)](#-how-rag-works-visual-explanation)
- [Prerequisites](#-prerequisites)
- [Project Structure](#-project-structure)
- [Setup Guide (Step by Step)](#-setup-guide-step-by-step)
- [Running the Application](#-running-the-application)
- [Understanding Each File](#-understanding-each-file)
- [How to Add Your Own Documents](#-how-to-add-your-own-documents)
- [Key Concepts Explained](#-key-concepts-explained)
- [Troubleshooting](#-troubleshooting)
- [Next Steps & Learning Resources](#-next-steps--learning-resources)

---

## 🎯 What This Project Does

This application lets you:

1. **Load your documents** (`.txt` and `.pdf` files)
2. **Store them in a local vector database** (ChromaDB)
3. **Ask questions** about your documents using natural language
4. **Get AI-generated answers** powered by Google Gemini, based on your documents

### Example

```
❓ Your question: What is a list in Python?

💡 ANSWER:
A list in Python is one of the most versatile data structures.
Lists are ordered collections. You can create a list like:
fruits = ["apple", "banana", "cherry"]... (answer from YOUR documents)

📄 SOURCES USED:
  Source 1: data/sample_docs/python_basics.txt
  Preview: Lists and List Comprehensions - Lists are one of Python's most versatile...
```

---

## 🔄 How RAG Works (Visual Explanation)

```
┌──────────────────────────────────────────────────────────────────┐
│                     DOCUMENT INGESTION (One-time Setup)          │
│                                                                  │
│  📄 Documents   →   ✂️ Split into    →   🔢 Create      →  💾 Store │
│  (.txt, .pdf)       small chunks        embeddings       in DB   │
│                     (500 chars)         (vectors)       (ChromaDB)│
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     QUESTION ANSWERING (Every Query)             │
│                                                                  │
│  ❓ Question  →  🔍 Search DB for   →  📝 Build prompt  →  🤖 Gemini │
│  "What is      relevant chunks       with context         generates│
│   a list?"     (similarity search)   + question           answer   │
└──────────────────────────────────────────────────────────────────┘
```

### The Flow in Simple Terms:

1. **You load documents** → The app reads your files and breaks them into small pieces ("chunks")
2. **Chunks become numbers** → Each chunk is converted into a list of numbers called an "embedding" (vector)
3. **Numbers are stored** → These embeddings are saved in a local database (ChromaDB)
4. **You ask a question** → Your question is also converted to an embedding
5. **Find similar chunks** → The database finds chunks whose embeddings are most similar to your question
6. **AI generates an answer** → The relevant chunks + your question are sent to Gemini, which writes an answer

---

## ✅ Prerequisites

Before you start, make sure you have:

### 1. Python 3.10 or higher

Check your Python version:

```bash
python3 --version
```

If you don't have Python, download it from [python.org](https://www.python.org/downloads/).

### 2. Google Gemini API Key (FREE)

You need a Gemini API key to use Google's AI. Here's how to get one:

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (it looks like `AIzaSy...`)

> 💡 **Is it free?** Yes! Google offers a free tier for Gemini API that is more than enough for learning and development.

### 3. pip (Python package manager)

This comes with Python. Verify it's installed:

```bash
pip3 --version
```

---

## 📁 Project Structure

```
python-rag-project/
│
├── main.py                      # 🚀 Entry point - run this file
│
├── src/                         # Source code (the brains)
│   ├── __init__.py              #    Makes src a Python package
│   ├── config.py                #    Configuration & settings
│   ├── document_loader.py       #    Load & split documents
│   ├── vector_store.py          #    Vector database operations
│   └── rag_chain.py             #    RAG pipeline (ties it all together)
│
├── data/
│   └── sample_docs/             # 📄 Put your documents here
│       ├── python_basics.txt    #    Sample doc about Python
│       └── ai_ml_basics.txt     #    Sample doc about AI/ML
│
├── docs/                        # 📖 Additional documentation
│   └── ARCHITECTURE.md          #    Detailed architecture guide
│
├── requirements.txt             # 📦 Python dependencies
├── .env.example                 # 🔑 Template for API key
├── .gitignore                   # 🚫 Files to exclude from Git
└── README.md                    # 📘 This file
```

---

## 🚀 Setup Guide (Step by Step)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/python-rag-project.git
cd python-rag-project
```

### Step 2: Create a Virtual Environment

A virtual environment keeps this project's dependencies separate from your system Python.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

> 🔍 **How do I know it's activated?** Your terminal prompt will show `(venv)` at the beginning.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all the libraries listed in `requirements.txt`. It may take a minute or two.

### Step 4: Set Up Your API Key

```bash
# Copy the example file to create your .env file
cp .env.example .env
```

Now open `.env` in your editor and replace the placeholder with your real API key:

```env
GOOGLE_API_KEY=AIzaSyYourActualKeyGoesHere
```

> ⚠️ **NEVER commit your `.env` file to GitHub!** The `.gitignore` file is already set up to prevent this.

### Step 5: Run the Application

```bash
python main.py
```

That's it! 🎉

---

## ▶️ Running the Application

### First Run

On first run, the app will automatically:
1. Load the sample documents from `data/sample_docs/`
2. Split them into chunks
3. Create embeddings and store them in ChromaDB
4. Start the interactive Q&A mode

```
╔══════════════════════════════════════════════════════════╗
║         🧠 Python RAG with Google Gemini                ║
║         Retrieval-Augmented Generation Demo             ║
╚══════════════════════════════════════════════════════════╝

📦 No vector database found. Starting document ingestion...

📥 DOCUMENT INGESTION
============================================================
Loading documents from data/sample_docs/...
  ✅ Loaded: python_basics.txt
  ✅ Loaded: ai_ml_basics.txt

📄 Split 2 documents into 15 chunks
   (chunk_size=500, overlap=100)

🔄 Creating embeddings and storing in vector database...
✅ Vector store created with 15 chunks!

🤖 RAG QUESTION-ANSWERING
============================================================
Ask me anything about the documents in the knowledge base!
Type 'quit' or 'exit' to stop.
```

### Sample Questions to Try

```
❓ What is a list in Python?
❓ How does error handling work in Python?
❓ What is RAG and why is it useful?
❓ What are the different types of machine learning?
❓ Explain what embeddings are
❓ What is the difference between supervised and unsupervised learning?
```

---

## 📖 Understanding Each File

### `main.py` — The Entry Point

**What it does:** Orchestrates the entire application. It calls the ingestion pipeline first, then starts the interactive Q&A loop.

**Key concepts:**
- `ingest_documents()` — Loads, chunks, and stores documents (run once)
- `interactive_query()` — The Q&A loop where you ask questions

---

### `src/config.py` — Configuration

**What it does:** Loads the API key from `.env` and defines all settings in one place.

**Key concepts:**
- `load_dotenv()` — Reads `.env` file into environment variables
- `os.getenv()` — Gets an environment variable's value
- Settings like `CHUNK_SIZE`, `TOP_K_RESULTS` control the RAG behavior

---

### `src/document_loader.py` — Document Loading & Chunking

**What it does:** Reads `.txt` and `.pdf` files, then splits them into overlapping chunks.

**Key concepts:**
- `TextLoader` / `PyPDFLoader` — Read different file formats
- `RecursiveCharacterTextSplitter` — Splits text intelligently (tries paragraphs first, then sentences, then words)
- **Chunk overlap** ensures no information is lost at boundaries

---

### `src/vector_store.py` — Vector Database

**What it does:** Converts text chunks into embeddings and stores them in ChromaDB for fast similarity search.

**Key concepts:**
- **Embeddings** — Text converted to numerical vectors
- **ChromaDB** — Local vector database (no cloud/account needed)
- **Similarity search** — Finding the most relevant chunks for a question

---

### `src/rag_chain.py` — The RAG Pipeline

**What it does:** Connects all the pieces: retriever + prompt template + Gemini LLM.

**Key concepts:**
- **PromptTemplate** — A template that structures how we ask the LLM
- **RetrievalQA** — A LangChain chain that automates the retrieve-then-generate flow
- **chain_type="stuff"** — Puts all retrieved chunks into one prompt (simplest approach)

---

## 📄 How to Add Your Own Documents

1. Place your `.txt` or `.pdf` files in the `data/sample_docs/` directory
2. Run the application: `python main.py`
3. When prompted, choose to re-ingest documents (type `y`)
4. The new documents will be processed and added to the knowledge base

### Supported File Types

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain Text | `.txt` | Simplest format, works great |
| PDF | `.pdf` | Extracts text from each page |

### Tips for Good Results

- **Be specific**: More focused documents give better answers
- **Use clear formatting**: Headers and paragraphs help the chunking process
- **No images-only PDFs**: The system extracts text, not images from PDFs
- **File size**: Keep individual files reasonable (under 50 pages for PDFs)

---

## 🔑 Key Concepts Explained

### What is an Embedding?

An embedding is a way to represent text as a list of numbers (called a "vector"). For example:

```
"Python is a programming language" → [0.12, -0.45, 0.78, 0.33, ...]
                                     (hundreds of numbers)
```

**Why?** Computers can't understand text directly, but they can compare numbers. Similar texts produce similar number patterns, so we can find relevant documents by comparing these patterns.

### What is a Vector Database?

A vector database is a specialized database designed to store and search through embeddings efficiently. Think of it like Google Search, but for your own documents. In this project, we use **ChromaDB** because:

- ✅ Runs completely locally (no cloud account needed)
- ✅ Free and open-source
- ✅ Perfect for learning and prototyping
- ✅ Stores data on your computer in the `chroma_db/` folder

### What is a Prompt Template?

A prompt template tells the AI how to use the retrieved information. Ours says:

```
"Here is some context from the knowledge base: [CHUNKS]
Based on this context, please answer: [QUESTION]"
```

This ensures the AI answers based on YOUR documents, not just its general knowledge.

### What is LangChain?

LangChain is a Python framework that makes it easy to build applications with LLMs. It provides ready-made components for:
- Loading documents
- Splitting text
- Managing embeddings
- Connecting to LLMs
- Building RAG pipelines

Think of it as "the glue" that connects all the pieces together.

---

## 🔧 Troubleshooting

### Error: "Google API Key not found!"

**Solution:** Make sure you:
1. Created a `.env` file (not `.env.example`)
2. Added your real API key to it
3. The key doesn't have quotes around it in the `.env` file

```env
# ✅ Correct
GOOGLE_API_KEY=AIzaSyYourKeyHere

# ❌ Wrong
GOOGLE_API_KEY="AIzaSyYourKeyHere"
```

### Error: "ModuleNotFoundError: No module named 'langchain'"

**Solution:** Make sure your virtual environment is activated and dependencies are installed:

```bash
source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt  # Install dependencies
```

### Error: "No documents found!"

**Solution:** Make sure you have `.txt` or `.pdf` files in the `data/sample_docs/` directory.

### Error: "API key not valid" or "Permission denied"

**Solution:**
1. Verify your API key is correct at [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Make sure the Gemini API is enabled for your Google Cloud project
3. Try creating a new API key

### Slow first run?

The first time you run the application, it needs to:
- Download embedding model data
- Process all documents into embeddings
- This is normal and only happens once. Subsequent runs will be much faster.

---

## 📚 Next Steps & Learning Resources

### Enhance This Project

Once you're comfortable with the basics, try these improvements:

1. **Add a web interface** — Use Streamlit or Gradio to create a web UI
2. **Support more file types** — Add support for `.docx`, `.csv`, or `.html`
3. **Add conversation memory** — Let the AI remember previous questions in the session
4. **Deploy to cloud** — Host the vector database on a cloud service
5. **Add more documents** — Build a specialized knowledge base for a specific topic

### Learning Resources

- [LangChain Documentation](https://python.langchain.com/docs/introduction/)
- [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [RAG Explained (Google blog)](https://cloud.google.com/use-cases/retrieval-augmented-generation)

---

## 🐙 Push to GitHub

If you want to host this project on GitHub, follow these steps:

### 1. Create a New Repository on GitHub

1. Go to [github.com/new](https://github.com/new)
2. Enter repository name: `python-rag-project`
3. Set it to **Public** or **Private**
4. **Do NOT** check "Add a README" (we already have one)
5. Click **"Create repository"**

### 2. Push Your Code

```bash
# Navigate to the project folder
cd python-rag-project

# Stage all files
git add .

# Create the first commit
git commit -m "Initial commit: Python RAG with Google Gemini"

# Connect to your GitHub repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/python-rag-project.git

# Push to GitHub
git push -u origin main
```

### 3. Verify

After pushing, verify on GitHub that:
- ✅ `.env` is **NOT** in the repository (contains your secret API key)
- ✅ `chroma_db/` is **NOT** in the repository (local database)
- ✅ `venv/` is **NOT** in the repository (virtual environment)
- ✅ `.env.example` **IS** in the repository (template for other users)

> ⚠️ **Security Note:** If you accidentally commit your API key, revoke it immediately at [Google AI Studio](https://aistudio.google.com/app/apikey) and create a new one.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

**Built with ❤️ using Python, Google Gemini, LangChain, and ChromaDB**
