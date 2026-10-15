from medrag.retrieval.bm25 import BM25Index
from medrag.retrieval.retriever import HybridRetriever
from medrag.store_pg import InMemoryChunkStore, chunk_row, row_chunk
from tests.fakes import FakeEmbedder, OverlapReranker, make_chunk

TEXTS = ["tinnitus is ringing in the ear", "sinusitis is inflammation of the sinuses", "the nasal septum has three parts"]


def store_of(chunks):
    return InMemoryChunkStore(chunks, FakeEmbedder().encode([c.text for c in chunks]))


def test_query_ranks_by_cosine():
    chunks = [make_chunk(i, t) for i, t in enumerate(TEXTS)]
    hits = store_of(chunks).query(FakeEmbedder().encode(["ringing ear tinnitus"])[0], n=2)
    assert len(hits) == 2 and hits[0][0] == chunks[0] and hits[0][1] >= hits[1][1]


def test_query_filters_by_document_and_caps_n():
    chunks = [make_chunk(0, TEXTS[0], doc="a"), make_chunk(1, TEXTS[1], doc="b")]
    s = store_of(chunks)
    q = FakeEmbedder().encode(["sinus"])[0]
    assert [c.doc_id for c, _ in s.query(q, n=10, doc_ids=["b"])] == ["b"]
    assert s.query(q, n=10, doc_ids=["zzz"]) == []
    assert len(s.query(q, n=10)) == 2


def test_empty_store():
    s = InMemoryChunkStore([], [])
    assert s.count() == 0 and s.all_chunks() == [] and s.documents() == {}
    assert s.query([0.1, 0.2], n=5) == []


def test_rows_roundtrip_with_title_and_user():
    c = make_chunk(3, TEXTS[0], doc="book-1", page=7, section="Ear > Tinnitus")
    c = type(c)(**{**c.__dict__, "topic": "Tinnitus"})
    r = chunk_row(c, [0.12345678, 1.0], user_id="alice")
    assert r["book_id"] == "book-1" and r["user_id"] == "alice" and r["embedding"] == [0.123457, 1.0]
    back = row_chunk(r, doc_title=c.doc_title)
    assert back == c
    s = InMemoryChunkStore.from_rows([r], "Book One")
    assert s.documents() == {"book-1": "Book One"} and s.count() == 1


def test_plugs_into_the_hybrid_retriever():
    chunks = [make_chunk(i, t) for i, t in enumerate(TEXTS)]
    retriever = HybridRetriever(store_of(chunks), FakeEmbedder(), BM25Index(chunks), OverlapReranker(), top_n=2)
    results = retriever.retrieve("what is tinnitus")
    assert results[0].chunk == chunks[0]
