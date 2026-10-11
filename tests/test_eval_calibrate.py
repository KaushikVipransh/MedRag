import pytest

from medrag.eval.calibrate import choose_threshold, rates, split_dev_test
from medrag.eval.consistency import consistency_scores, cosine, jaccard
from medrag.eval.gold import GoldItem
from medrag.types import Answer, Citation
from tests.fakes import FakeEmbedder, make_chunk


def test_split_is_deterministic_and_complete():
    items = [GoldItem(f"in-{i:03d}", "in_scope", "q", "answer") for i in range(40)]
    dev, test = split_dev_test(items)
    assert split_dev_test(items) == (dev, test)
    assert len(dev) + len(test) == 40 and dev and test


def test_rates():
    assert rates([0.9, 0.5], [0.1, 0.6], 0.55) == (0.5, 0.5)
    assert rates([], [], 0.5) == (None, None)


def test_choose_threshold_maximises_rejection_under_pass_constraint():
    scores_in, scores_out = [0.9, 0.95, 0.99], [0.001, 0.2]
    assert choose_threshold(scores_in, scores_out, min_pass=1.0) == 0.9
    assert choose_threshold(scores_in, scores_out, min_pass=0.6) == 0.95


def test_jaccard_and_cosine():
    assert jaccard(set(), set()) == 1.0 and jaccard({"1"}, {"2"}) == 0.0
    assert jaccard({"1", "2"}, {"2"}) == 0.5
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_consistency_scores():
    c = make_chunk(0, "t", page=9)
    same = Answer("Tinnitus is ringing [1].", refused=False, citations=[Citation(1, c)])
    other = Answer("Sinusitis is inflammation.", refused=False)
    scores = consistency_scores([[same, same, same], [same, same, other]], FakeEmbedder().encode)
    assert scores["per_question"][0] == {"text_cosine": pytest.approx(1.0), "citation_jaccard": 1.0}
    assert scores["per_question"][1]["citation_jaccard"] == pytest.approx(1 / 3)
    assert scores["text_cosine"] < 1.0
