from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from medrag.types import Answer, Citation, RetrievedChunk
from tests.fakes import make_chunk


class FakePipeline:
    def __init__(self):
        chunk = make_chunk(0, "Tinnitus is ringing in the ear.", page=159, section="Ear > Tinnitus")
        self._answer = Answer(
            text="Tinnitus is ringing in the ear [1].",
            refused=False,
            citations=[Citation(1, chunk)],
            chunks=[RetrievedChunk(chunk, score=0.9, rerank_score=0.9)],
            standalone_question="What is tinnitus?",
        )
        self.retriever = SimpleNamespace(store=SimpleNamespace(documents=lambda: {"book-a": "Book A"}))
        self.questions = []

    def answer(self, question, history=None, doc_ids=None):
        self.questions.append((question, history, doc_ids))
        return self._answer


@pytest.fixture
def app(monkeypatch):
    import medrag.pipeline

    fake = FakePipeline()
    monkeypatch.setattr(medrag.pipeline, "build_pipeline", lambda settings: fake)
    import streamlit as st

    st.cache_resource.clear()
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    return AppTest.from_file(str(app_path), default_timeout=30), fake


def test_question_shows_answer_and_source(app):
    at, fake = app
    at.run()
    at.chat_input[0].set_value("What is tinnitus?").run()
    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "Tinnitus is ringing in the ear [1]." in rendered
    assert any("p. 160" in e.label for e in at.expander)
    assert fake.questions[0][0] == "What is tinnitus?"
    assert fake.questions[0][2] == ["book-a"]
