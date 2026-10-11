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


def test_item_can_be_recategorised_with_an_answer_key(tmp_path, monkeypatch):
    path = tmp_path / "gold.jsonl"
    save_gold(path, [GoldItem("near-002", "near_miss", "Amoxicillin dose?", "partial_or_refuse")])
    monkeypatch.setenv("MEDRAG_GOLD_PATH", str(path))
    monkeypatch.setenv("MEDRAG_EVIDENCE_PATH", str(tmp_path / "e.jsonl"))
    monkeypatch.setenv("MEDRAG_CHROMA_DIR", str(tmp_path / "chroma"))
    import streamlit as st

    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key="category").set_value("in_scope")
    at.selectbox(key="expected").set_value("answer")
    at.text_area(key="answer_key").input("40 mg/kg/day in three divided doses.")
    at.text_input(key="pages").input("57")
    at.button(key="verify").click().run()
    item = load_gold(path)[0]
    assert item.category == "in_scope" and item.answer_key.startswith("40 mg") and item.gold_pages == ["57"]


def test_in_scope_item_without_answer_key_is_not_saved(gold):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="answer_key").input("")
    at.button(key="verify").click().run()
    assert load_gold(gold)[0].verified is False
    assert any("answer key" in e.value.lower() for e in at.error)
