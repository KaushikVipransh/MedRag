import pytest

from medrag.eval.report import render_report, summarize_system
from medrag.eval.runner import ItemResult


def r(id, cat, **kw):
    return ItemResult(id=id, category=cat, expected="answer", system="medrag", question="q", **kw)


RESULTS = [
    r("i1", "in_scope", correctness="correct", book_support=1.0, citation_support=1.0, latency_s=1.0),
    r("i2", "in_scope", correctness="partial", book_support=0.5, latency_s=3.0),
    r("i3", "in_scope", refused=True, refusal_reason="low_relevance", latency_s=2.0),
    r("i4", "in_scope", refused=True, refusal_reason="llm_error", judge_errors=["system llm_error"]),
    r("o1", "out_of_scope", refused=True, handled_ok=True),
    r("o2", "out_of_scope", handled_ok=False),
    r("n1", "near_miss", handled_ok=True),
    r("a1", "adversarial", handled_ok=None, judge_errors=["handling: bad"]),
]


def test_summarize_system():
    s = summarize_system(RESULTS)
    assert s["n_in_scope"] == 3 and s["llm_errors"] == 1
    assert s["answer_rate"] == pytest.approx(2 / 3)
    assert s["correct"] == 0.5 and s["correct_or_partial"] == 1.0
    assert s["correct_overall"] == pytest.approx(1 / 3)
    assert s["book_support"] == 0.75 and s["citation_support"] == 1.0
    assert s["oos_refusal"] == 0.5 and s["near_miss_ok"] == 1.0 and s["adversarial_ok"] is None
    assert s["median_latency_s"] == 2.0


def test_render_report_marks_drafts_and_formats_values():
    meta = {"date": "2026-10-12", "generator": "g", "judge": "j", "unverified_included": True,
            "gold_counts": {"in_scope": 3}}
    md = render_report(meta, systems={"medrag": summarize_system(RESULTS)},
                       retrieval={"rrf_rerank": {"recall@1": 0.5, "recall@5": 1.0, "recall@10": 1.0,
                                                 "mrr": 0.75, "ndcg@5": 0.8, "n": 2}})
    assert md.startswith("> **DRAFT**")
    assert "| medrag |" in md and "66.7%" in md and "—" in md
    assert "| rrf_rerank |" in md


def test_render_report_without_draft_banner():
    md = render_report({"date": "d", "generator": "g", "judge": "j", "unverified_included": False, "gold_counts": {}})
    assert not md.startswith("> **DRAFT**") and md.startswith("# MedRAG evaluation")
