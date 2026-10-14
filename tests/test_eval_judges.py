from medrag.eval.baseline import NoRagBaseline
from medrag.eval.judges import format_sources, judge_correctness, judge_handling, judge_support, support_scores
from medrag.generation.llm import LLMError
from medrag.types import RetrievedChunk
from tests.fakes import FakeLLM, make_chunk

CHUNKS = [RetrievedChunk(make_chunk(0, "Tinnitus is ringing in the ear.", page=145), score=0.9)]


def test_format_sources():
    assert format_sources(CHUNKS) == "[1] (page 146)\nTinnitus is ringing in the ear."


def test_correctness_valid():
    j = judge_correctness(FakeLLM(['{"verdict": "partial", "reason": "misses x"}']), "q", "key", "ans")
    assert j.ok and j.data["verdict"] == "partial"


def test_correctness_invalid_verdict_is_an_error():
    j = judge_correctness(FakeLLM(['{"verdict": "great"}']), "q", "key", "ans")
    assert not j.ok and "verdict" in j.error


def test_correctness_non_json_and_llm_error():
    assert not judge_correctness(FakeLLM(["I think it is correct"]), "q", "k", "a").ok
    j = judge_correctness(FakeLLM(error=LLMError("429")), "q", "k", "a")
    assert not j.ok and "429" in j.error


def test_support_and_scores():
    llm = FakeLLM(['{"claims": [{"claim": "a", "cited": [1], "supported_by_cited": true, "supported_by_any": true},'
                   ' {"claim": "b", "cited": [], "supported_by_cited": false, "supported_by_any": false}]}'])
    j = judge_support(llm, "a [1]. b.", CHUNKS)
    assert j.ok
    assert support_scores(j.data["claims"]) == (0.5, 1.0)
    assert support_scores([]) == (None, None)


def test_support_without_passages_is_an_error():
    assert not judge_support(FakeLLM(), "answer", []).ok


def test_support_wrong_schema():
    assert not judge_support(FakeLLM(['{"claims": "none"}']), "a", CHUNKS).ok


def test_handling():
    j = judge_handling(FakeLLM(['{"appropriate": false, "reason": "accepted false premise"}']), "q", "a", CHUNKS)
    assert j.ok and j.data["appropriate"] is False
    assert not judge_handling(FakeLLM(['{"appropriate": "yes"}']), "q", "a", CHUNKS).ok


def test_baseline_answer_refusal_and_error():
    assert NoRagBaseline(FakeLLM(["Tinnitus is ringing."])).answer("q").text == "Tinnitus is ringing."
    refused = NoRagBaseline(FakeLLM(["NOT_FOUND"])).answer("q")
    assert refused.refused and refused.refusal_reason == "not_in_context"
    failed = NoRagBaseline(FakeLLM(error=LLMError("down"))).answer("q")
    assert failed.refused and failed.refusal_reason == "llm_error"
    empty = NoRagBaseline(FakeLLM(["  "])).answer("q")
    assert empty.refusal_reason == "llm_error"


def test_support_claims_with_non_boolean_flags_are_rejected():
    bad = '{"claims": [{"claim": "a", "cited": [1], "supported_by_cited": "false", "supported_by_any": "false"}]}'
    assert not judge_support(FakeLLM([bad]), "a [1].", CHUNKS).ok


def test_support_claims_that_are_not_objects_are_rejected():
    assert not judge_support(FakeLLM(['{"claims": ["a claim"]}']), "a.", CHUNKS).ok
