import pytest

from medrag.eval.gold import GoldItem
from medrag.eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank, score_retriever
from medrag.types import RetrievedChunk
from tests.fakes import make_chunk

PAGES = ["10", "12", "10", "15"]


def test_recall_at_k():
    assert recall_at_k(PAGES, {"12"}, 1) == 0.0
    assert recall_at_k(PAGES, {"12"}, 2) == 1.0


def test_reciprocal_rank_uses_unique_pages():
    assert reciprocal_rank(PAGES, {"15"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(PAGES, {"99"}) == 0.0


def test_ndcg():
    assert ndcg_at_k(PAGES, {"12"}, 3) == pytest.approx(0.6309, abs=1e-4)
    assert ndcg_at_k(PAGES, {"10", "15"}, 4) == pytest.approx(1.5 / 1.6309, abs=1e-4)
    assert ndcg_at_k(PAGES, {"99"}, 4) == 0.0


def test_mean_ignores_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([]) is None and mean([None]) is None


class FixedRetriever:
    def __init__(self, pages):
        self.pages = pages

    def retrieve(self, query, doc_ids=None):
        return [RetrievedChunk(make_chunk(i, "t", page=int(p) - 1), score=1.0) for i, p in enumerate(self.pages)]


def test_score_retriever_skips_items_without_gold_pages():
    items = [
        GoldItem("a", "in_scope", "q1", "answer", gold_pages=["12"]),
        GoldItem("b", "in_scope", "q2", "answer", gold_pages=[]),
    ]
    scores = score_retriever(FixedRetriever(["10", "12"]), items)
    assert scores["n"] == 1
    assert scores["recall@1"] == 0.0 and scores["recall@5"] == 1.0 and scores["mrr"] == 0.5


def test_all_metrics_rank_unique_pages_consistently():
    pages, gold = ["191", "172", "191", "155", "191"], {"155"}
    assert recall_at_k(pages, gold, 3) == 1.0  # third unique page
    assert reciprocal_rank(pages, gold) == pytest.approx(1 / 3)
    assert ndcg_at_k(pages, gold, 3) == pytest.approx(0.5)
