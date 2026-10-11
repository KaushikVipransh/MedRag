from medrag.types import Answer, Chunk
from tests.fakes import make_chunk


def test_chunk_metadata_roundtrip():
    chunk = make_chunk(3, "Tinnitus is ringing in the ear.", page=159, section="Ear > Tinnitus")
    rebuilt = Chunk.from_record(chunk.id, chunk.text, chunk.metadata())
    assert rebuilt == chunk


def test_chunk_metadata_has_no_none_values():
    chunk = make_chunk(0, "text", section="")
    assert all(v is not None for v in chunk.metadata().values())


def test_answer_defaults():
    answer = Answer(text="hi", refused=False)
    assert answer.citations == [] and answer.chunks == [] and answer.warnings == []
    assert answer.refusal_reason is None and answer.emergency is False
