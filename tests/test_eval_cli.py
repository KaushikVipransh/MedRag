import json

import pytest

from medrag.eval.__main__ import main
from medrag.eval.gold import GoldItem, save_gold


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDRAG_GOLD_PATH", str(tmp_path / "gold.jsonl"))
    monkeypatch.setenv("MEDRAG_EVIDENCE_PATH", str(tmp_path / "evidence.jsonl"))
    monkeypatch.setenv("MEDRAG_EVAL_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("MEDRAG_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("MEDRAG_EVAL_CACHE", str(tmp_path / "cache"))
    return tmp_path


def test_draft_refuses_to_overwrite_existing_gold(env):
    save_gold(env / "gold.jsonl", [GoldItem("in-001", "in_scope", "q", "answer", verified=True)])
    with pytest.raises(SystemExit, match="--force"):
        main(["draft"])


def test_run_requires_verified_items(env):
    save_gold(env / "gold.jsonl", [GoldItem("in-001", "in_scope", "q", "answer")])
    with pytest.raises(SystemExit, match="review_gold.py"):
        main(["run"])


def test_report_writes_markdown(env):
    results = env / "results"
    results.mkdir()
    meta = {"date": "2026-10-12", "generator": "g", "judge": "j", "unverified_included": True, "gold_counts": {"in_scope": 1}}
    (results / "retrieval.json").write_text(json.dumps({"meta": meta, "variants": {
        "rrf_rerank": {"recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0, "ndcg@5": 1.0, "n": 1}}}))
    main(["report"])
    reports = list((env / "reports").glob("*evaluation-DRAFT.md"))
    assert len(reports) == 1 and "| rrf_rerank |" in reports[0].read_text(encoding="utf-8")


def test_all_requires_verified_items(env):
    save_gold(env / "gold.jsonl", [GoldItem("in-001", "in_scope", "q", "answer")])
    with pytest.raises(SystemExit, match="review_gold.py"):
        main(["all"])


def test_ids_file_restricts_items(env):
    import argparse

    from medrag.eval.__main__ import _items
    from medrag.eval.config import EvalSettings

    save_gold(env / "gold.jsonl", [GoldItem("in-001", "in_scope", "q1", "answer", verified=True),
                                   GoldItem("in-002", "in_scope", "q2", "answer", verified=True)])
    ids = env / "ids.txt"
    ids.write_text("# dev subset\nin-002\n")
    args = argparse.Namespace(include_unverified=False, ids_file=str(ids))
    assert [i.id for i in _items(EvalSettings.from_env(), args)] == ["in-002"]
