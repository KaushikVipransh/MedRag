"""Rendering shared by the local and multi-user apps."""
import streamlit as st

from medrag.safety import EMERGENCY_MESSAGE


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


def render_message_body(msg: dict) -> None:
    """Everything inside one chat bubble: emergency banner, text, warnings and sources."""
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
