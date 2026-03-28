from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────
PDF_SOURCE   = "pdfs/"          # folder path OR single .pdf file path
INDEX_PATH   = "faiss_index"    # where the FAISS index is saved/loaded
EMBED_MODEL  = "nomic-embed-text"
LLM_MODEL    = "gemma2:2b"
TOP_K        = 4                # how many chunks to retrieve


# ─────────────────────────────────────────────
# STEP 1a — LOAD PDF(s)
# ─────────────────────────────────────────────
def load_documents(source: str):
    """
    Accepts either:
      - a single PDF file path  → "paper.pdf"
      - a folder of PDFs        → "pdfs/"
    Returns a list of LangChain Document objects.
    Each Document has:
      .page_content  → the text
      .metadata      → {"source": "...", "page": N}
    """
    if os.path.isfile(source):
        print(f"📄 Loading single file: {source}")
        loader = PyPDFLoader(source)

    elif os.path.isdir(source):
        print(f"📁 Loading all PDFs from folder: {source}")
        loader = DirectoryLoader(
            source,
            glob="**/*.pdf",        # finds PDFs in sub-folders too
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
    else:
        raise FileNotFoundError(f"'{source}' is not a valid file or folder.")

    docs = loader.load()
    print(f"   Loaded {len(docs)} page(s) from {len(set(d.metadata.get('source') for d in docs))} file(s)\n")
    return docs


# ─────────────────────────────────────────────
# STEP 1b — TEXT SPLITTING
# ─────────────────────────────────────────────
def split_documents(docs):
    """
    Splits each page's text into overlapping chunks so:
      - No chunk exceeds the embedding model's token limit
      - Sentences that fall on a boundary are still captured
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)  # preserves metadata (page number, source)
    print(f"📦 Split into {len(chunks)} chunks\n")
    return chunks


# ─────────────────────────────────────────────
# STEP 1c & 1d — EMBEDDINGS + VECTOR STORE
# ─────────────────────────────────────────────
def build_or_load_vectorstore(chunks, index_path: str):
    """
    If a saved FAISS index exists → load it (skips re-embedding, much faster).
    Otherwise → embed all chunks and save the index for next time.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(index_path):
        print(f"⚡ Loading existing FAISS index from '{index_path}' ...")
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True   # safe because WE created this file
        )
        print("   Index loaded.\n")
    else:
        print("🔨 Building FAISS index (this may take a minute) ...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_path)
        print(f"   Index built and saved to '{index_path}'\n")

    return vectorstore


# ─────────────────────────────────────────────
# STEP 2 — RETRIEVER
# ─────────────────────────────────────────────
def build_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ─────────────────────────────────────────────
# STEP 3 — LLM
# ─────────────────────────────────────────────
def build_llm():
    return ChatOllama(model=LLM_MODEL)


# ─────────────────────────────────────────────
# STEP 4 — PROMPT TEMPLATE
# ─────────────────────────────────────────────
def build_prompt():
    return PromptTemplate(
        template="""
You are a helpful research assistant. 
Answer ONLY using the provided context from the PDF(s).
If the context does not contain enough information, say "I don't know based on the provided documents."

--- CONVERSATION HISTORY ---
{history}

--- CONTEXT FROM PDF ---
{context}

Question: {question}
Answer:
        """,
        input_variables=["history", "context", "question"],
    )


# ─────────────────────────────────────────────
# HELPER — format retrieved Document objects
# ─────────────────────────────────────────────
def format_docs(docs):
    """
    Joins retrieved chunks into a single string.
    Also shows the source file and page number for each chunk.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[Source: {os.path.basename(source)}, Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_history(history: list[dict]) -> str:
    """
    Converts the conversation history list into a readable string for the prompt.
    history = [{"role": "user"/"assistant", "content": "..."}]
    """
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        prefix = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# STEP 5 — CHAIN
# ─────────────────────────────────────────────
def build_chain(prompt, llm):
    return prompt | llm | StrOutputParser()


# ─────────────────────────────────────────────
# STEP 6 — CHAT LOOP (with memory)
# ─────────────────────────────────────────────
def chat_loop(retriever, chain):
    """
    Interactive Q&A loop.
    Keeps a rolling conversation history (last 6 exchanges = 12 messages)
    so the LLM has context for follow-up questions.
    """
    print("=" * 60)
    print("  📚 PDF Research Assistant  (type 'quit' to exit)")
    print("=" * 60)
    print("Tip: Ask follow-up questions — I remember the conversation!\n")

    history = []   # list of {"role": ..., "content": ...}

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        # Retrieve relevant chunks
        retrieved_docs = retriever.invoke(question)
        context        = format_docs(retrieved_docs)

        # Build history string (keep last 6 turns to avoid prompt bloat)
        history_str = format_history(history[-6:])

        # Run the RAG chain
        answer = chain.invoke({
            "history":  history_str,
            "context":  context,
            "question": question,
        })

        print(f"\nAssistant: {answer}")

        # Show which pages were used
        sources = set(
            f"{os.path.basename(d.metadata.get('source','?'))} p.{d.metadata.get('page','?')}"
            for d in retrieved_docs
        )
        print(f"📎 Sources used: {', '.join(sources)}\n")

        # Save to history
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # 1. Load

    raw_docs = load_documents("Optimized Deep Learning Model for Pneumonia Classification using Chest X-Rays.pdf")

    # 2. Split
    chunks = split_documents(raw_docs)

    # 3. Vectorstore (loads from disk if already built)
    vectorstore = build_or_load_vectorstore(chunks, INDEX_PATH)

    # 4. Retriever
    retriever = build_retriever(vectorstore)

    # 5. LLM + Prompt + Chain
    llm     = build_llm()
    prompt  = build_prompt()
    chain   = build_chain(prompt, llm)

    # 6. Chat
    chat_loop(retriever, chain)


if __name__ == "__main__":
    main()