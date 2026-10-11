import pytest

from medrag.retrieval.bm25 import BM25Index
from medrag.retrieval.retriever import HybridRetriever
from medrag.store import ChunkStore
from tests.fakes import FakeEmbedder, FixedReranker, OverlapReranker, make_chunk

CHUNKS = [
    make_chunk(0, "Tinnitus is a ringing sound in the ear originating within the patient."),
    make_chunk(1, "Mondini dysplasia is incomplete partition of the cochlea."),
    make_chunk(2, "Acute sinusitis is treated with amoxicillin and decongestants."),
    make_chunk(3, "The nasal septum consists of columellar, membranous and septum proper parts."),
    make_chunk(4, "Otosclerosis causes conductive hearing loss.", doc="book-b"),
]


@pytest.fixture
def parts(tmp_path):
    store = ChunkStore(tmp_path / "db", "test")
    emb = FakeEmbedder()
    store.add(CHUNKS, emb.encode([c.text for c in CHUNKS]))
    return store, emb, BM25Index(CHUNKS)


def test_returns_top_n_unique_chunks(parts):
    store, emb, bm25 = parts
    results = HybridRetriever(store, emb, bm25, OverlapReranker(), top_n=3).retrieve("ear tinnitus ringing")
    ids = [r.chunk.id for r in results]
    assert len(ids) == 3 and len(set(ids)) == 3


def test_exact_rare_term_ranks_first(parts):
    store, emb, bm25 = parts
    results = HybridRetriever(store, emb, bm25, OverlapReranker()).retrieve("Mondini")
    assert results[0].chunk.id == CHUNKS[1].id


def test_reranker_decides_final_order(parts):
    store, emb, bm25 = parts
    reranker = FixedReranker({"nasal septum": 0.9, "Tinnitus": 0.1})
    results = HybridRetriever(store, emb, bm25, reranker).retrieve("tinnitus septum")
    assert results[0].chunk.id == CHUNKS[3].id
    assert results[0].rerank_score == 0.9 and results[0].score == 0.9
    assert results[0].rrf_score > 0


def test_without_reranker_orders_by_rrf(parts):
    store, emb, bm25 = parts
    results = HybridRetriever(store, emb, bm25, None).retrieve("sinusitis amoxicillin")
    assert results[0].chunk.id == CHUNKS[2].id
    assert all(r.rerank_score is None and r.score == r.rrf_score for r in results)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_doc_filter(parts):
    store, emb, bm25 = parts
    results = HybridRetriever(store, emb, bm25, OverlapReranker()).retrieve("hearing loss", doc_ids=["book-b"])
    assert results and all(r.chunk.doc_id == "book-b" for r in results)
