import json

import pytest

from medrag.eval.gold import GoldItem
from medrag.eval.runner import evaluate_item, load_results, run_system
from medrag.types import Answer, Citation, RetrievedChunk
from tests.fakes import FakeLLM, make_chunk

CHUNK = make_chunk(0, "Tinnitus is ringing in the ear.", page=159)
RC = RetrievedChunk(CHUNK, score=0.9, rerank_score=0.9)
CORRECT = '{"verdict": "correct", "reason": "ok"}'
SUPPORTED = '{"claims": [{"claim": "ringing", "cited": [1], "supported_by_cited": true, "supported_by_any": true}]}'
IN = GoldItem("in-001", "in_scope", "What is tinnitus?", "answer", answer_key="Ringing in the ear.", gold_pages=["160"])
OOS = GoldItem("oos-001", "out_of_scope", "COVID vaccine?", "refuse")
NEAR = GoldItem("near-001", "near_miss", "Appendicitis symptoms?", "partial_or_refuse")


class Answerer:
    def __init__(self, answer: Answer | None = None, boom: bool = False):
        self._answer, self.boom = answer, boom

    def answer(self, question, history=None, doc_ids=None):
        if self.boom:
            raise AssertionError("should not be called")
        return self._answer


def no_passages(q):
    return []


def test_in_scope_answer_is_judged():
    ans = Answer("Tinnitus is ringing [1].", refused=False, citations=[Citation(1, CHUNK)], chunks=[RC])
    r = evaluate_item(IN, "medrag", Answerer(ans), FakeLLM([CORRECT, SUPPORTED]), no_passages)
    assert r.correctness == "correct" and r.book_support == 1.0 and r.citation_support == 1.0
    assert r.retrieved_pages == ["160"] and r.cited_pages == ["160"] and r.top_rerank == 0.9
    assert r.judge_errors == []


def test_baseline_without_chunks_is_checked_against_retrieved_passages():
    judge = FakeLLM([CORRECT, '{"claims": [{"claim": "x", "cited": [], "supported_by_cited": false, "supported_by_any": true}]}'])
    r = evaluate_item(IN, "no_rag", Answerer(Answer("Ringing.", refused=False)), judge, lambda q: [RC])
    assert r.book_support == 1.0 and r.citation_support is None
    assert "Tinnitus is ringing in the ear." in judge.calls[1][1]["content"]


def test_refused_in_scope_is_not_judged():
    judge = FakeLLM()
    r = evaluate_item(IN, "medrag", Answerer(Answer("Not found", refused=True, refusal_reason="low_relevance")), judge, no_passages)
    assert r.refused and r.correctness is None and judge.calls == []


def test_out_of_scope_scored_by_refusal():
    refused = evaluate_item(OOS, "medrag", Answerer(Answer("x", refused=True, refusal_reason="low_relevance")), FakeLLM(), no_passages)
    answered = evaluate_item(OOS, "no_rag", Answerer(Answer("Two doses.", refused=False)), FakeLLM(), no_passages)
    assert refused.handled_ok is True and answered.handled_ok is False


def test_near_miss_answer_goes_to_handling_judge():
    judge = FakeLLM(['{"appropriate": false, "reason": "outside knowledge"}'])
    r = evaluate_item(NEAR, "medrag", Answerer(Answer("Fever and pain.", refused=False, chunks=[RC])), judge, no_passages)
    assert r.handled_ok is False


def test_malformed_judge_output_is_recorded_not_raised():
    ans = Answer("Ringing [1].", refused=False, citations=[Citation(1, CHUNK)], chunks=[RC])
    r = evaluate_item(IN, "medrag", Answerer(ans), FakeLLM(["not json", "also not json"]), no_passages)
    assert r.correctness is None and r.book_support is None and len(r.judge_errors) == 2


def test_system_llm_error_is_recorded_without_judging():
    judge = FakeLLM()
    r = evaluate_item(IN, "medrag", Answerer(Answer("down", refused=True, refusal_reason="llm_error")), judge, no_passages)
    assert r.judge_errors == ["system llm_error"] and judge.calls == []


def test_run_system_writes_and_resumes(tmp_path):
    out = tmp_path / "medrag.jsonl"
    ans = Answer("x", refused=True, refusal_reason="low_relevance")
    first = run_system([OOS, IN], "medrag", Answerer(ans), FakeLLM(), no_passages, out, progress=lambda m: None)
    assert len(first) == 2 and len(out.read_text().splitlines()) == 2
    again = run_system([OOS, IN], "medrag", Answerer(boom=True), FakeLLM(), no_passages, out, progress=lambda m: None)
    assert [r.id for r in again] == ["oos-001", "in-001"]
    assert load_results(out)[0].handled_ok is True


def test_quota_exhaustion_aborts_the_run_without_saving_the_item(tmp_path):
    from medrag.eval.llm_wrappers import QuotaExhausted

    class Exhausted:
        def answer(self, question, history=None, doc_ids=None):
            raise QuotaExhausted("daily quota")

    out = tmp_path / "medrag.jsonl"
    with pytest.raises(QuotaExhausted):
        run_system([IN], "medrag", Exhausted(), FakeLLM(), no_passages, out, progress=lambda m: None)
    assert load_results(out) == []


def test_rows_with_errors_are_retried_on_resume(tmp_path):
    out = tmp_path / "medrag.jsonl"
    failed = Answer("down", refused=True, refusal_reason="llm_error")
    run_system([OOS], "medrag", Answerer(failed), FakeLLM(), no_passages, out, progress=lambda m: None)
    ok = Answer("x", refused=True, refusal_reason="low_relevance")
    again = run_system([OOS], "medrag", Answerer(ok), FakeLLM(), no_passages, out, progress=lambda m: None)
    assert again[0].handled_ok is True and again[0].judge_errors == []


def test_edited_item_is_re_evaluated(tmp_path):
    out = tmp_path / "medrag.jsonl"
    refused = Answer("x", refused=True, refusal_reason="low_relevance")
    run_system([OOS], "medrag", Answerer(refused), FakeLLM(), no_passages, out, progress=lambda m: None)
    edited = GoldItem("oos-001", "out_of_scope", "A different question?", "refuse")
    answered = Answer("Two doses.", refused=False)
    again = run_system([edited], "medrag", Answerer(answered), FakeLLM(), no_passages, out, progress=lambda m: None)
    assert again[0].question == "A different question?" and again[0].handled_ok is False


def test_config_change_invalidates_saved_rows(tmp_path):
    out = tmp_path / "medrag.jsonl"
    refused = Answer("x", refused=True, refusal_reason="low_relevance")
    run_system([OOS], "medrag", Answerer(refused), FakeLLM(), no_passages, out, config="threshold=0.01",
               progress=lambda m: None)
    answered = Answer("Two doses.", refused=False)
    again = run_system([OOS], "medrag", Answerer(answered), FakeLLM(), no_passages, out, config="threshold=0.12",
                       progress=lambda m: None)
    assert again[0].handled_ok is False
