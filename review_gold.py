"""Review UI for the evaluation gold set: streamlit run review_gold.py"""
import streamlit as st

from medrag.config import Settings
from medrag.eval.config import EvalSettings
from medrag.eval.gold import CATEGORIES, EXPECTED, load_evidence, load_gold, parse_pages, save_gold

st.set_page_config(page_title="MedRAG gold review", page_icon="✅", layout="wide")
cfg = EvalSettings.from_env()


@st.cache_resource
def chunk_lookup() -> dict:
    try:
        from medrag.store import ChunkStore

        settings = Settings.from_env()
        return {c.id: c for c in ChunkStore(settings.chroma_dir, settings.collection).all_chunks()}
    except Exception:
        return {}


items = load_gold(cfg.gold_path)
evidence = load_evidence(cfg.evidence_path)

st.title("Gold set review")
if not items:
    st.warning(f"No gold items at {cfg.gold_path}. Run: python -m medrag.eval draft")
    st.stop()

done = sum(i.verified for i in items)
st.progress(done / len(items), text=f"{done} / {len(items)} verified")
with st.sidebar:
    show_all = st.toggle("Include already-verified items")
    for cat in CATEGORIES:
        group = [i for i in items if i.category == cat]
        st.caption(f"{cat}: {sum(i.verified for i in group)} / {len(group)} verified")

pending = [i for i in items if show_all or not i.verified]
if not pending:
    st.success("Every item is verified. Run the evaluation without --include-unverified.")
    st.stop()

idx = st.session_state.get("idx", 0) % len(pending)
item = pending[idx]
st.subheader(f"{item.id} · {item.category}" + (" · verified" if item.verified else ""))

ev = evidence.get(item.id)
if ev:
    st.info(f"Evidence (p. {', '.join(item.gold_pages)} · {item.chapter}): “{ev['quote']}”")
    chunk = chunk_lookup().get(ev.get("chunk_id"))
    if chunk:
        with st.expander("Full source passage"):
            st.text(chunk.text)
if item.category == "in_scope":
    st.caption("Check: the answer key is correct per the book, the question is answerable from it, "
               "and the gold pages are where the answer is printed.")
else:
    st.caption("Check: the book really does not answer this (search it if unsure) and the expected behaviour fits. "
               "If the book does answer it, change the category to in_scope and add an answer key and pages.")

FIELDS = ("question", "category", "expected", "answer_key", "pages", "notes")


def next_item(advance: bool = False) -> None:
    """Forget the widget values (they are keyed per field, not per item) and move on."""
    for key in FIELDS:
        st.session_state.pop(key, None)
    if advance:
        st.session_state.idx = idx + 1
    st.rerun()


with st.form(f"form-{item.id}"):
    question = st.text_area("Question", item.question, key="question")
    category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(item.category), key="category")
    expected = st.selectbox("Expected behaviour", EXPECTED, index=EXPECTED.index(item.expected), key="expected")
    answer_key = st.text_area("Answer key (required for in_scope)", item.answer_key, key="answer_key")
    pages = st.text_input("Gold printed pages (comma-separated)", ", ".join(item.gold_pages), key="pages")
    notes = st.text_input("Notes", item.notes, key="notes")
    c1, c2, c3 = st.columns(3)
    verify = c1.form_submit_button("✅ Verify", key="verify")
    reject = c2.form_submit_button("🗑 Reject", key="reject")
    skip = c3.form_submit_button("Skip", key="skip")

if verify:
    gold_pages, unknown = parse_pages(pages, {c.page_label for c in chunk_lookup().values()})
    if category == "in_scope" and not answer_key.strip():
        st.error("An in-scope item needs an answer key.")
    elif unknown:
        st.error(f"Unknown page label(s): {', '.join(unknown)}. Use printed page numbers exactly as the book shows them.")
    else:
        item.question, item.category, item.expected = question.strip(), category, expected
        item.answer_key, item.notes, item.gold_pages = answer_key.strip(), notes.strip(), gold_pages
        item.verified = True
        save_gold(cfg.gold_path, items)
        next_item()
elif reject:
    save_gold(cfg.gold_path, [i for i in items if i.id != item.id])
    next_item()
elif skip:
    next_item(advance=True)
