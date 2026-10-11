import pytest

from medrag.eval.variants import BM25Only, DenseOnly, V1Merge, build_variants
from medrag.retrieval.bm25 import BM25Index
from medrag.store import ChunkStore
from tests.fakes import FakeEmbedder, OverlapReranker, make_chunk

CHUNKS = [
    make_chunk(0, "Tinnitus is a ringing sound in the ear originating within the patient."),
    make_chunk(1, "Mondini dysplasia is incomplete partition of the cochlea."),
    make_chunk(2, "Sinusitis, acute or chronic, is treated with amoxicillin."),
    make_chunk(3, "X" * 80 + " alpha tail"),
    make_chunk(4, "X" * 80 + " beta tail"),
]


@pytest.fixture
def parts(tmp_path):
    store = ChunkStore(tmp_path / "db", "test")
    emb = FakeEmbedder()
    store.add(CHUNKS, emb.encode([c.text for c in CHUNKS]))
    return store, emb, BM25Index(CHUNKS)


def test_dense_only(parts):
    store, emb, _ = parts
    results = DenseOnly(store, emb, top_n=2).retrieve("ringing ear tinnitus")
    assert len(results) == 2 and results[0].chunk.id == CHUNKS[0].id


def test_bm25_only(parts):
    _, _, bm25 = parts
    assert BM25Only(bm25).retrieve("Mondini")[0].chunk.id == CHUNKS[1].id


def test_v1_merge_dedupes_on_first_80_chars_and_sorts(parts):
    store, emb, _ = parts
    results = V1Merge(store, emb, CHUNKS, top_n=10).retrieve("alpha beta tail")
    keys = [r.chunk.text[:80] for r in results]
    assert len(keys) == len(set(keys))
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_v1_merge_keeps_v1_case_sensitive_bm25(parts):
    store, emb, _ = parts
    v1 = V1Merge(store, emb, CHUNKS)
    assert v1._bm25_hits("sinusitis") == []  # v1 split() tokens: "Sinusitis," never matches


def test_build_variants_keys(parts):
    store, emb, bm25 = parts
    variants = build_variants(store, emb, bm25, CHUNKS, OverlapReranker())
    assert set(variants) == {"dense", "bm25", "v1_merge", "rrf", "rrf_rerank"}
    assert variants["rrf"].reranker is None and variants["rrf_rerank"].top_n == 10
