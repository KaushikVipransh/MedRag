from medrag.embeddings import Embedder
from medrag.retrieval.bm25 import BM25Index
from medrag.retrieval.fusion import reciprocal_rank_fusion
from medrag.retrieval.rerank import Reranker
from medrag.store import ChunkStore
from medrag.types import Chunk, RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        bm25: BM25Index,
        reranker: Reranker | None,
        candidates: int = 30,
        top_n: int = 5,
    ):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker
        self.candidates = candidates
        self.top_n = top_n

    def retrieve(self, query: str, doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        dense = self.store.query(self.embedder.encode([query])[0], self.candidates, doc_ids)
        sparse = self.bm25.search(query, self.candidates, doc_ids)

        by_id: dict[str, Chunk] = {c.id: c for c, _ in dense}
        by_id.update({c.id: c for c, _ in sparse})
        fused = reciprocal_rank_fusion([[c.id for c, _ in dense], [c.id for c, _ in sparse]])
        ordered = sorted(fused, key=fused.get, reverse=True)[: self.candidates]
        if not ordered:
            return []

        if self.reranker is None:
            return [
                RetrievedChunk(by_id[i], score=fused[i], rrf_score=fused[i])
                for i in ordered[: self.top_n]
            ]

        scores = self.reranker.score(query, [by_id[i].text for i in ordered])
        results = [
            RetrievedChunk(by_id[i], score=s, rrf_score=fused[i], rerank_score=s)
            for i, s in zip(ordered, scores)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: self.top_n]
