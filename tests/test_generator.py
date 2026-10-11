from medrag.generation.generator import Generator
from medrag.generation.llm import LLMError
from medrag.generation.prompts import NOT_FOUND_MESSAGE
from medrag.types import RetrievedChunk
from tests.fakes import FakeLLM, make_chunk


def chunks():
    return [
        RetrievedChunk(make_chunk(0, "Tinnitus is ringing in the ear.", page=159), score=0.9, rerank_score=0.9),
        RetrievedChunk(make_chunk(1, "Maskers are worn like hearing aids.", page=160), score=0.8, rerank_score=0.8),
    ]


def test_grounded_answer_with_citations():
    llm = FakeLLM(["Tinnitus is ringing in the ear [1]. Maskers can help [2]."])
    ans = Generator(llm).generate("What is tinnitus?", chunks())
    assert not ans.refused and ans.refusal_reason is None
    assert [c.marker for c in ans.citations] == [1, 2]
    assert ans.citations[0].chunk.page_label == "160"
    assert ans.warnings == []


def test_prompt_contains_sources_and_question():
    llm = FakeLLM(["Tinnitus is ringing [1]."])
    Generator(llm, temperature=0.1).generate("What is tinnitus?", chunks())
    system, user = llm.calls[0]
    assert system["role"] == "system" and "NOT_FOUND" in system["content"]
    assert "[1] (Book A, page 160" in user["content"]
    assert "Question: What is tinnitus?" in user["content"]


def test_invalid_citation_is_removed_and_flagged():
    llm = FakeLLM(["Tinnitus is a ringing sound heard by the patient [1][7]."])
    ans = Generator(llm).generate("q", chunks())
    assert "[7]" not in ans.text
    assert [c.marker for c in ans.citations] == [1]
    assert any("non-existent" in w for w in ans.warnings)


def test_uncited_sentence_is_flagged():
    llm = FakeLLM(["Tinnitus is a ringing sound that originates inside the patient. Maskers help [2]."])
    ans = Generator(llm).generate("q", chunks())
    assert any("without a citation" in w for w in ans.warnings)


def test_answer_with_no_citations_is_flagged():
    llm = FakeLLM(["Tinnitus is ringing."])
    ans = Generator(llm).generate("q", chunks())
    assert not ans.refused
    assert "answer cites no sources" in ans.warnings


def test_model_refusal():
    ans = Generator(FakeLLM(["NOT_FOUND"])).generate("What is the COVID schedule?", chunks())
    assert ans.refused and ans.refusal_reason == "not_in_context"
    assert ans.text == NOT_FOUND_MESSAGE
    assert ans.citations == []


def test_llm_failure_becomes_friendly_answer():
    ans = Generator(FakeLLM(error=LLMError("RateLimitError: slow down"))).generate("q", chunks())
    assert ans.refused and ans.refusal_reason == "llm_error"
    assert "try again" in ans.text.lower()
    assert any("RateLimitError" in w for w in ans.warnings)


def test_empty_llm_output_is_an_llm_error():
    ans = Generator(FakeLLM(["   "])).generate("q", chunks())
    assert ans.refused and ans.refusal_reason == "llm_error"
