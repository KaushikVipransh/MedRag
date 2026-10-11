import pytest

from medrag.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_beats_single_list_leaders():
    scores = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    assert max(scores, key=scores.get) == "b"


def test_exact_scores():
    scores = reciprocal_rank_fusion([["a"], ["a", "b"]], k=60)
    assert scores["a"] == pytest.approx(2 / 61)
    assert scores["b"] == pytest.approx(1 / 62)


def test_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == {}
