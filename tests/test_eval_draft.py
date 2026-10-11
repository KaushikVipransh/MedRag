from medrag.eval.draft import draft_gold, draft_item, parse_json_object, quote_in_passage, sample_chunks
from medrag.eval.fixed_sets import fixed_items
from medrag.generation.llm import LLMError
from tests.fakes import FakeLLM, make_chunk

S = "Tinnitus is a ringing sound in the ear originating within the patient."
FILLER = " It is described in detail in this chapter of the textbook." * 8
OK = '{"question": "What is tinnitus?", "answer_key": "A ringing sound arising within the patient.", "evidence_quote": "%s"}' % S


def long_chunk(i, section):
    return make_chunk(i, S + FILLER, page=100 + i, section=section)


def test_parse_json_object():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object("no json here") is None
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object("{broken") is None


def test_quote_in_passage():
    assert quote_in_passage("tinnitus IS a ringing   sound in the ear", S)
    assert not quote_in_passage("Tinnitus is a buzzing noise in the ear", S)
    assert not quote_in_passage("ringing", S)


def test_sample_chunks_spreads_across_chapters_and_skips_front_matter():
    chunks = [long_chunk(10 * c + i, f"Sec > Chapter {c}") for c in range(3) for i in range(4)]
    chunks += [long_chunk(99, "Preface"), make_chunk(98, "short", section="Sec > Chapter 0")]
    picked = sample_chunks(chunks, 3)
    assert {c.section for c in picked} == {"Sec > Chapter 0", "Sec > Chapter 1", "Sec > Chapter 2"}
    everything = sample_chunks(chunks, 100)
    assert len(everything) == 12 and all(c.section != "Preface" and c.text != "short" for c in everything)


def test_draft_item_statuses():
    chunk = long_chunk(1, "Sec > Ear")
    assert draft_item(FakeLLM([OK]), chunk)[1] == "ok"
    paraphrased = OK.replace("ringing sound in the ear", "buzzing noise in the head")
    assert draft_item(FakeLLM([paraphrased]), chunk) == (None, "bad_quote")
    assert draft_item(FakeLLM(['{"skip": true}']), chunk) == (None, "skip")
    assert draft_item(FakeLLM(["garbage"]), chunk) == (None, "bad_json")
    assert draft_item(FakeLLM(error=LLMError("down")), chunk) == (None, "llm_error")


def test_draft_gold_builds_items_and_evidence():
    chunks = [long_chunk(i, f"Sec > Chapter {i}") for i in range(3)]
    items, evidence, stats = draft_gold(FakeLLM([OK, '{"skip": true}', OK]), chunks, n_target=2)
    assert [i.id for i in items] == ["in-001", "in-002"]
    assert all(i.category == "in_scope" and i.expected == "answer" and not i.verified for i in items)
    assert all(len(i.gold_pages) == 1 and i.chapter.startswith("Sec > Chapter") for i in items)
    assert set(evidence) == {"in-001", "in-002"} and evidence["in-001"]["quote"] == S
    assert stats == {"ok": 2, "skip": 1}


def test_fixed_items():
    items = fixed_items()
    by_cat = {c: [i for i in items if i.category == c] for c in ("out_of_scope", "near_miss", "adversarial")}
    assert [len(v) for v in by_cat.values()] == [25, 10, 6]
    assert len({i.id for i in items}) == len(items)
    assert all(i.expected == "refuse" for i in by_cat["out_of_scope"])
    assert all(not i.verified for i in items)
