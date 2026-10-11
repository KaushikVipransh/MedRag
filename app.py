import json
import time
from pathlib import Path

import streamlit as st

from medrag import pipeline as pipeline_module
from medrag.config import Settings
from medrag.safety import DISCLAIMER, EMERGENCY_MESSAGE

FEEDBACK_PATH = Path("data/feedback.jsonl")

st.set_page_config(page_title="MedRAG", page_icon="⚕️", layout="wide")


@st.cache_resource(show_spinner="Loading models and index…")
def get_pipeline():
    return pipeline_module.build_pipeline(Settings.from_env())


def log_feedback(idx: int) -> None:
    msg = st.session_state.messages[idx]
    answer = msg["answer"]
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": msg["question"],
        "standalone_question": answer.standalone_question,
        "answer": answer.text,
        "refusal_reason": answer.refusal_reason,
        "cited_chunks": [c.chunk.id for c in answer.citations],
        "thumbs_up": st.session_state.get(f"fb_{idx}") == 1,
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def render_sources(answer) -> None:
    cited = {c.chunk.id for c in answer.citations}
    for n, rc in enumerate(answer.chunks, start=1):
        c = rc.chunk
        status = "cited" if c.id in cited else "not cited"
        label = f"[{n}] {c.doc_title} — p. {c.page_label} · {c.section or 'section unknown'} ({status})"
        with st.expander(label):
            st.markdown(c.text)
            if st.session_state.get("debug"):
                st.caption(f"rerank={rc.rerank_score}  rrf={rc.rrf_score:.4f}  id={c.id}")


def render_message(idx: int, msg: dict) -> None:
    with st.chat_message(msg["role"]):
        answer = msg.get("answer")
        if answer is not None and answer.emergency:
            st.error(EMERGENCY_MESSAGE)
        st.markdown(msg["content"])
        if answer is None:
            return
        if not answer.refused and not answer.citations:
            st.warning("This answer cites no sources. Verify it against the book before relying on it.")
        if st.session_state.get("debug"):
            if answer.standalone_question and answer.standalone_question != msg["question"]:
                st.caption(f"Searched for: {answer.standalone_question}")
            for warning in answer.warnings:
                st.caption(f"⚠ {warning}")
        if answer.chunks:
            render_sources(answer)
        if answer.refusal_reason != "invalid_input":
            st.feedback("thumbs", key=f"fb_{idx}", on_change=log_feedback, args=(idx,))


st.title("⚕️ MedRAG")
st.caption("Answers grounded in the indexed medical textbook, with page citations you can check.")
st.info(DISCLAIMER)

try:
    rag = get_pipeline()
except Exception as exc:
    st.error(f"MedRAG could not start: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    documents = rag.retriever.store.documents()
    selected = st.multiselect(
        "Search in", options=list(documents), default=list(documents), format_func=documents.get
    )
    st.toggle("Show retrieval details", key="debug")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for i, message in enumerate(st.session_state.messages):
    render_message(i, message)

if prompt := st.chat_input("Ask a question about the book…"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Searching the book…"):
        try:
            answer = rag.answer(prompt, history=history, doc_ids=selected or None)
        except Exception as exc:
            st.session_state.messages.pop()
            st.error(f"Something went wrong while answering: {exc}")
            st.stop()
    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "answer": answer, "question": prompt}
    )
    st.rerun()
