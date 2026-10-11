import pytest

from medrag.store import ChunkStore
from tests.fakes import FakeEmbedder, make_chunk


@pytest.fixture
def store(tmp_path):
    return ChunkStore(tmp_path / "db", "test")


def _add(store, chunks):
    store.add(chunks, FakeEmbedder().encode([c.text for c in chunks]))


def test_add_and_query_roundtrip(store):
    chunks = [
        make_chunk(0, "tinnitus is ringing in the ear"),
        make_chunk(1, "sinusitis is inflammation of the sinuses"),
        make_chunk(2, "the nasal septum has three parts"),
    ]
    _add(store, chunks)
    assert store.count() == 3
    hits = store.query(FakeEmbedder().encode(["ringing ear tinnitus"])[0], n=2)
    assert len(hits) == 2
    assert hits[0][0] == chunks[0]
    assert hits[0][1] >= hits[1][1]


def test_query_filters_by_document(store):
    _add(store, [make_chunk(0, "ear wax", doc="book-a"), make_chunk(0, "ear wax removal", doc="book-b")])
    hits = store.query(FakeEmbedder().encode(["ear wax"])[0], n=5, doc_ids=["book-b"])
    assert [h[0].doc_id for h in hits] == ["book-b"]


def test_delete_document(store):
    _add(store, [make_chunk(0, "a", doc="book-a"), make_chunk(0, "b", doc="book-b")])
    store.delete_document("book-a")
    assert [c.doc_id for c in store.all_chunks()] == ["book-b"]


def test_documents_lists_titles(store):
    _add(store, [make_chunk(0, "a", doc="book-a"), make_chunk(1, "b", doc="book-a")])
    assert store.documents() == {"book-a": "Book A"}


def test_query_on_empty_store(store):
    assert store.query(FakeEmbedder().encode(["x"])[0], n=5) == []
