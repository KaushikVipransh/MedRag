import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
import os
from dotenv import load_dotenv


# --- PAGE CONFIG ---
st.set_page_config(page_title="MedRAG Assistant", page_icon="⚕️", layout="wide")

# --- CONFIG & CONSTANTS ---

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "medical_book"
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 10

SYSTEM_PROMPT = """You are a professional medical research assistant.
Rules:
1. Answer ONLY from the provided context.
2. Always cite the page number, e.g.: "(Source: Page 142)"
3. If not in context, say: "This information was not found in the source book."
4. Be concise and professional."""

# --- LOAD RESOURCES (Cached to prevent reloading on every click) ---
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer(EMBED_MODEL)
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection(COLLECTION)
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    # Build BM25
    all_docs = collection.get(include=["documents", "metadatas"])
    bm25_corpus = [doc.split() for doc in all_docs["documents"]]
    bm25_index = BM25Okapi(bm25_corpus)
    
    return embedder, collection, groq_client, bm25_index, all_docs

embedder, collection, groq_client, bm25_index, all_docs = load_resources()

# --- HYBRID SEARCH LOGIC ---
def hybrid_search(query):
    q_emb = embedder.encode(query).tolist()
    sr = collection.query(query_embeddings=[q_emb], n_results=TOP_K * 2, include=["documents", "metadatas", "distances"])
    bm25s = bm25_index.get_scores(query.split())
    top_b = sorted(range(len(bm25s)), key=lambda i: bm25s[i], reverse=True)[:TOP_K * 2]

    seen, merged = set(), []
    for doc, meta, dist in zip(sr["documents"][0], sr["metadatas"][0], sr["distances"][0]):
        k = doc[:80]
        if k not in seen:
            seen.add(k)
            merged.append({"text": doc, "page_number": meta.get("page_number", "?"), "score": 1 - dist})
    
    for idx in top_b:
        k = all_docs["documents"][idx][:80]
        if k not in seen:
            seen.add(k)
            merged.append({"text": all_docs["documents"][idx], 
                           "page_number": all_docs["metadatas"][idx].get("page_number", "?"), 
                           "score": bm25s[idx] / (max(bm25s) + 1e-9)})

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:TOP_K]

# --- STREAMLIT UI ---
st.title("⚕️ MedRAG — Medical Research Navigator")
st.caption("AI-powered medical literature search using Hybrid RAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a medical question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical database..."):
            chunks = hybrid_search(prompt)
            context = "\n\n---\n\n".join(f"[Page {c['page_number']}]\n{c['text']}" for c in chunks)
            
            try:
                resp = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                    ],
                    temperature=0.1
                )
                answer = resp.choices[0].message.content
                
                # Format sources
                pages = sorted(set(str(c["page_number"]) for c in chunks if c["page_number"] != "?"))
                full_response = f"{answer}\n\n---\n📚 **Sources:** Page(s) {', '.join(pages)}"
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                st.error(f"Error connecting to Groq: {e}")

# Sidebar info
with st.sidebar:
    st.header("Database Stats")
    st.metric("Indexed Chunks", f"{len(all_docs['documents']):,}")
    st.info("Using LLaMA 3.3-70B via Groq")