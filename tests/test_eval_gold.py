import pytest

from medrag.eval.gold import GoldItem, load_evidence, load_gold, save_evidence, save_gold, select


def item(i, cat="in_scope", verified=True, expected="answer"):
    return GoldItem(id=f"x-{i}", category=cat, question=f"q{i}", expected=expected, gold_pages=["10"], verified=verified)


def test_roundtrip(tmp_path):
    items = [item(1), GoldItem("oos-001", "out_of_scope", "Ménière?", "refuse")]
    save_gold(tmp_path / "g.jsonl", items)
    assert load_gold(tmp_path / "g.jsonl") == items


def test_missing_file_is_empty(tmp_path):
    assert load_gold(tmp_path / "none.jsonl") == []


def test_invalid_category_rejected():
    with pytest.raises(ValueError):
        GoldItem("a", "bogus", "q", "answer")


def test_invalid_expected_rejected():
    with pytest.raises(ValueError):
        GoldItem("a", "in_scope", "q", "maybe")


def test_select_filters_verified_category_and_limit():
    items = [item(1), item(2, verified=False), item(3), item(4, cat="out_of_scope", expected="refuse")]
    assert [i.id for i in select(items, include_unverified=False)] == ["x-1", "x-3", "x-4"]
    assert [i.id for i in select(items, include_unverified=True, categories={"in_scope"})] == ["x-1", "x-2", "x-3"]
    assert [i.id for i in select(items, include_unverified=True, limit_per_category=1)] == ["x-1", "x-4"]


def test_evidence_roundtrip(tmp_path):
    ev = {"in-001": {"chunk_id": "c1", "quote": "Tinnitus is ringing."}}
    save_evidence(tmp_path / "e.jsonl", ev)
    assert load_evidence(tmp_path / "e.jsonl") == ev
    assert load_evidence(tmp_path / "missing.jsonl") == {}


def test_parse_pages_flags_unknown_labels():
    from medrag.eval.gold import parse_pages

    assert parse_pages(" 146, 147 ,", {"146", "147"}) == (["146", "147"], [])
    assert parse_pages("146-147, p. 146", {"146", "147"}) == (["146-147", "p. 146"], ["146-147", "p. 146"])
    assert parse_pages("146", set()) == (["146"], [])  # no index available: nothing to check against
