from rank_bm25 import BM25Okapi

from medrag.retrieval.retriever import HybridRetriever
from medrag.types import Chunk, RetrievedChunk


class DenseOnly:
    def __init__(self, store, embedder, top_n: int = 10):
        self.store, self.embedder, self.top_n = store, embedder, top_n

    def retrieve(self, query: str, doc_ids=None) -> list[RetrievedChunk]:
        hits = self.store.query(self.embedder.encode([query])[0], self.top_n, doc_ids)
        return [RetrievedChunk(c, score=s) for c, s in hits]


class BM25Only:
    def __init__(self, bm25, top_n: int = 10):
        self.bm25, self.top_n = bm25, top_n

    def retrieve(self, query: str, doc_ids=None) -> list[RetrievedChunk]:
        return [RetrievedChunk(c, score=s) for c, s in self.bm25.search(query, self.top_n, doc_ids)]


class V1Merge:
    """Faithful reproduction of the v1 merge: dense similarity (1 - distance) and BM25
    scores divided by their max are sorted together as if comparable, BM25 uses raw
    str.split() tokens, and duplicates are detected by the first 80 characters."""

    def __init__(self, store, embedder, chunks: list[Chunk], top_n: int = 10):
        self.store, self.embedder, self.chunks, self.top_n = store, embedder, chunks, top_n
        self._bm25 = BM25Okapi([c.text.split() for c in chunks]) if chunks else None

    def _bm25_hits(self, query: str) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.split())
        top = max(scores) if len(scores) else 0.0
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_n * 2]
        return [(self.chunks[i], scores[i] / (top + 1e-9)) for i in order if scores[i] > 0]

    def retrieve(self, query: str, doc_ids=None) -> list[RetrievedChunk]:
        dense = self.store.query(self.embedder.encode([query])[0], self.top_n * 2, doc_ids)
        seen, merged = set(), []
        for chunk, score in [*dense, *self._bm25_hits(query)]:
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            key = chunk.text[:80]
            if key in seen:
                continue
            seen.add(key)
            merged.append(RetrievedChunk(chunk, score=float(score)))
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[: self.top_n]


def build_variants(store, embedder, bm25, chunks, reranker, top_n: int = 10) -> dict:
    return {
        "dense": DenseOnly(store, embedder, top_n),
        "bm25": BM25Only(bm25, top_n),
        "v1_merge": V1Merge(store, embedder, chunks, top_n),
        "rrf": HybridRetriever(store, embedder, bm25, None, top_n=top_n),
        "rrf_rerank": HybridRetriever(store, embedder, bm25, reranker, top_n=top_n),
    }
