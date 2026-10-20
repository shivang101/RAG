import os
import shutil

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion_pipeline import main as run_ingestion

load_dotenv()

PERSIST_DIR = "db/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="centered")


# ---------- Cached resources (loaded once per session) ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner="Loading vector store...")
def get_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model=LLM_MODEL, temperature=0)


def build_prompt(query, docs):
    context = "\n".join(f"- {d.page_content}" for d in docs)
    return f"""Based on the following documents, please answer this question: {query}

Documents:
{context}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""


def stream_answer(query, docs):
    """Yield the LLM answer token by token."""
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=build_prompt(query, docs)),
    ]
    for chunk in get_llm().stream(messages):
        if chunk.content:
            yield chunk.content


def show_sources(docs):
    with st.expander(f"📄 Sources ({len(docs)} chunks)"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}** — `{d.metadata.get('source', 'unknown')}`")
            st.text(d.page_content)
            st.divider()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("Vector store")

    if os.path.exists(PERSIST_DIR):
        st.success("Index found")
        if st.button("🔄 Rebuild index", use_container_width=True):
            with st.spinner("Rebuilding index from docs/ ..."):
                get_vectorstore.clear()
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)
                try:
                    run_ingestion()
                    st.success("Index rebuilt")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        st.warning("No index yet")
        if st.button("⚙️ Build index from docs/", use_container_width=True):
            with st.spinner("Ingesting documents..."):
                try:
                    run_ingestion()
                    get_vectorstore.clear()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- Main chat ----------
st.title("📚 Chat with your documents")
st.caption("Retrieval-Augmented Generation · ChromaDB + MiniLM + Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("docs"):
            show_sources(msg["docs"])

query = st.chat_input("Ask something about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if not os.path.exists(PERSIST_DIR):
            st.error("Vector store not found. Build the index from the sidebar first.")
        else:
            with st.spinner("Searching documents..."):
                retriever = get_vectorstore().as_retriever(search_kwargs={"k": k})
                docs = retriever.invoke(query)

            answer = st.write_stream(stream_answer(query, docs))
            show_sources(docs)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "docs": docs}
            )