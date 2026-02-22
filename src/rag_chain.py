"""
rag_chain.py - The RAG Pipeline (Putting It All Together)
==========================================================
This file creates the complete RAG (Retrieval-Augmented Generation) chain.

THE RAG FLOW:
┌─────────────┐    ┌───────────────┐    ┌─────────────┐    ┌──────────────┐
│  User asks   │───▶│  Find relevant │───▶│  Build prompt│───▶│  Gemini LLM  │
│  a question  │    │  chunks (DB)   │    │  with context│    │  generates   │
│              │    │               │    │              │    │  answer      │
└─────────────┘    └───────────────┘    └─────────────┘    └──────────────┘

WHAT IS A "CHAIN"?
In LangChain, a "chain" is a sequence of steps connected together.
Our chain: Question → Retrieve Context → Format Prompt → Generate Answer

WHAT IS A PROMPT TEMPLATE?
A prompt template is like a form letter with blanks to fill in.
We define the structure of what we want to tell the LLM, and then
fill in the blanks with the retrieved context and user's question.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.config import GOOGLE_API_KEY, LLM_MODEL, TOP_K_RESULTS
from src.vector_store import load_vector_store


# ─── The Prompt Template ─────────────────────────────────────
# This tells the LLM how to behave and what to do with the context
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Use ONLY the context below to answer the question
- If the context doesn't contain enough information, say "I don't have enough information in my knowledge base to answer this question."
- Be concise but thorough in your answers
- If relevant, mention which topic area the information comes from

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def create_rag_chain():
    """
    Build the complete RAG chain.

    This connects all the pieces:
    1. Vector Store (for retrieving relevant chunks)
    2. Prompt Template (for formatting the input to the LLM)
    3. Gemini LLM (for generating the final answer)

    Returns:
        A RetrievalQA chain that can answer questions

    Example:
        chain = create_rag_chain()
        result = chain.invoke({"query": "What is a list in Python?"})
        print(result["result"])
    """
    # Step 1: Load the vector store (our knowledge base)
    print("📚 Loading knowledge base...")
    vector_store = load_vector_store()

    # Step 2: Create the retriever
    # A retriever wraps the vector store and handles searching
    retriever = vector_store.as_retriever(
        search_type="similarity",       # Use cosine similarity
        search_kwargs={"k": TOP_K_RESULTS}  # Return top K results
    )

    # Step 3: Create the prompt template
    # This formats the context and question into a clear prompt
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Step 4: Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,  # Lower = more focused/deterministic answers
        convert_system_message_to_human=True,
    )

    # Step 5: Build the chain
    # RetrievalQA automatically:
    #   - Takes the user's question
    #   - Uses the retriever to find relevant chunks
    #   - Stuffs the chunks into the prompt template as "context"
    #   - Sends the completed prompt to the LLM
    #   - Returns the LLM's answer
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = put all retrieved docs into one prompt
        retriever=retriever,
        return_source_documents=True,  # Also return the source chunks used
        chain_type_kwargs={"prompt": prompt},
    )

    print("✅ RAG chain ready!\n")

    return rag_chain


def ask_question(chain, question: str) -> dict:
    """
    Ask a question to the RAG system.

    Args:
        chain: The RAG chain from create_rag_chain()
        question: The user's question as a string

    Returns:
        A dictionary with:
        - "result": The generated answer
        - "source_documents": The chunks used to generate the answer

    Example:
        chain = create_rag_chain()
        answer = ask_question(chain, "What is a dictionary in Python?")
        print(answer["result"])
    """
    # Invoke the chain with the question
    response = chain.invoke({"query": question})

    return response
