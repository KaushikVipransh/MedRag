from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from medrag.eval.gold import GoldItem, load_gold, save_evidence, save_gold

APP = str(Path(__file__).resolve().parent.parent / "review_gold.py")


@pytest.fixture
def gold(tmp_path, monkeypatch):
    path = tmp_path / "gold.jsonl"
    save_gold(path, [
        GoldItem("in-001", "in_scope", "What is tinnitus?", "answer", answer_key="Ringing.", gold_pages=["146"]),
        GoldItem("in-002", "in_scope", "What is otosclerosis?", "answer", answer_key="Bone disease.", gold_pages=["90"]),
    ])
    save_evidence(tmp_path / "evidence.jsonl", {"in-001": {"chunk_id": "c", "quote": "Tinnitus is ringing."}})
    monkeypatch.setenv("MEDRAG_GOLD_PATH", str(path))
    monkeypatch.setenv("MEDRAG_EVIDENCE_PATH", str(tmp_path / "evidence.jsonl"))
    monkeypatch.setenv("MEDRAG_CHROMA_DIR", str(tmp_path / "chroma"))
    import streamlit as st

    st.cache_resource.clear()
    return path


def test_verify_marks_item_and_saves(gold):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Tinnitus is ringing." in i.value for i in at.info)
    at.button(key="verify").click().run()
    items = load_gold(gold)
    assert items[0].verified is True and items[1].verified is False


def test_reject_removes_item(gold):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key="reject").click().run()
    assert [i.id for i in load_gold(gold)] == ["in-002"]
