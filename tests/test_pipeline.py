import pytest

from medrag.config import Settings
from medrag.generation.generator import Generator
from medrag.pipeline import MedRAG, build_pipeline
from medrag.retrieval.bm25 import BM25Index
from medrag.retrieval.retriever import HybridRetriever
from medrag.store import ChunkStore
from tests.fakes import FakeEmbedder, FakeLLM, FixedReranker, OverlapReranker, make_chunk

CHUNKS = [
    make_chunk(0, "Tinnitus is a ringing sound in the ear originating within the patient.", page=159),
    make_chunk(1, "Tinnitus maskers are worn like hearing aids.", page=160),
    make_chunk(2, "Acute sinusitis is treated with amoxicillin and decongestants.", page=229),
]
SETTINGS = Settings(rerank_threshold=0.01, top_n=3)


def make_rag(tmp_path, llm, reranker=None):
    store = ChunkStore(tmp_path / "db", "test")
    emb = FakeEmbedder()
    store.add(CHUNKS, emb.encode([c.text for c in CHUNKS]))
    retriever = HybridRetriever(store, emb, BM25Index(CHUNKS), reranker or OverlapReranker(), top_n=3)
    return MedRAG(retriever, Generator(llm), llm, SETTINGS)


def test_happy_path(tmp_path):
    llm = FakeLLM(["Tinnitus is a ringing sound originating within the patient [1]."])
    ans = make_rag(tmp_path, llm).answer("What is tinnitus?")
    assert not ans.refused
    assert "Tinnitus" in ans.citations[0].chunk.text
    assert ans.standalone_question == "What is tinnitus?"
    assert ans.emergency is False


def test_invalid_input_skips_everything(tmp_path):
    llm = FakeLLM()
    ans = make_rag(tmp_path, llm).answer("   ")
    assert ans.refused and ans.refusal_reason == "invalid_input"
    assert llm.calls == []


def test_low_relevance_refuses_without_calling_llm(tmp_path):
    llm = FakeLLM()
    ans = make_rag(tmp_path, llm, reranker=FixedReranker({})).answer("What is the COVID-19 vaccine schedule?")
    assert ans.refused and ans.refusal_reason == "low_relevance"
    assert llm.calls == []


def test_question_without_searchable_words_does_not_crash(tmp_path):
    ans = make_rag(tmp_path, FakeLLM()).answer("???")
    assert ans.refused and ans.refusal_reason == "invalid_input"


def test_follow_up_uses_condensed_question(tmp_path):
    llm = FakeLLM(["How are tinnitus maskers worn?", "They are worn like hearing aids [1]."])
    history = [
        {"role": "user", "content": "What is tinnitus?"},
        {"role": "assistant", "content": "A ringing sound [1]."},
    ]
    ans = make_rag(tmp_path, llm).answer("how are maskers worn?", history=history)
    assert ans.standalone_question == "How are tinnitus maskers worn?"
    assert "Question: How are tinnitus maskers worn?" in llm.calls[1][1]["content"]


def test_emergency_flag_is_set(tmp_path):
    llm = FakeLLM(["NOT_FOUND"])
    ans = make_rag(tmp_path, llm).answer("I can't breathe, what do I do about tinnitus?")
    assert ans.emergency is True


def test_build_pipeline_fails_fast_on_empty_index(tmp_path):
    with pytest.raises(RuntimeError, match="python -m medrag.ingest"):
        build_pipeline(Settings(chroma_dir=tmp_path / "empty"))


def test_question_with_only_stopwords_is_refused_deterministically(tmp_path):
    llm = FakeLLM()
    rag = make_rag(tmp_path, llm, reranker=FixedReranker({"Tinnitus": 0.9}))
    for q in ("what is it?", "1"):
        ans = rag.answer(q)
        assert ans.refused and ans.refusal_reason == "invalid_input"
    assert llm.calls == []


def test_stopword_follow_up_with_history_is_condensed_not_refused(tmp_path):
    llm = FakeLLM(["What is tinnitus?", "Tinnitus is a ringing sound [1]."])
    history = [{"role": "user", "content": "Tell me about tinnitus"}, {"role": "assistant", "content": "Sure [1]."}]
    ans = make_rag(tmp_path, llm).answer("what is it?", history=history)
    assert not ans.refused


def test_relevance_gate_decides_refusal(tmp_path):
    store = ChunkStore(tmp_path / "db", "test")
    emb = FakeEmbedder()
    store.add(CHUNKS, emb.encode([c.text for c in CHUNKS]))
    llm = FakeLLM()
    retriever = HybridRetriever(store, emb, BM25Index(CHUNKS), OverlapReranker(), top_n=3, gate=FixedReranker({}))
    ans = MedRAG(retriever, Generator(llm), llm, SETTINGS).answer("What is tinnitus?")
    assert ans.refused and ans.refusal_reason == "low_relevance" and ans.relevance == 0.0
    assert llm.calls == []


def test_answer_records_relevance(tmp_path):
    ans = make_rag(tmp_path, FakeLLM(["Tinnitus is a ringing sound [1]."])).answer("What is tinnitus?")
    assert ans.relevance == 1.0
