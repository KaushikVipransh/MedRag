from medrag.embeddings import Embedder
from medrag.retrieval.bm25 import BM25Index
from medrag.retrieval.fusion import reciprocal_rank_fusion
from medrag.retrieval.rerank import Reranker
from medrag.store import ChunkStore
from medrag.types import Chunk, RetrievedChunk, context_text


def rerank_text(chunk: Chunk) -> str:
    return context_text(chunk)


class HybridRetriever:
    """Dense + BM25 candidates fused with RRF, ordered by `reranker`. An optional `gate`
    cross-encoder scores the whole candidate pool; its best score is the question's
    relevance, used to decide whether the book can answer at all."""

    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        bm25: BM25Index,
        reranker: Reranker | None,
        candidates: int = 30,
        top_n: int = 5,
        gate: Reranker | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker
        self.candidates = candidates
        self.top_n = top_n
        self.gate = gate

    def retrieve(self, query: str, doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        return self.retrieve_with_relevance(query, doc_ids)[0]

    def retrieve_with_relevance(
        self, query: str, doc_ids: list[str] | None = None
    ) -> tuple[list[RetrievedChunk], float | None]:
        dense = self.store.query(self.embedder.encode([query])[0], self.candidates, doc_ids)
        sparse = self.bm25.search(query, self.candidates, doc_ids)

        by_id: dict[str, Chunk] = {c.id: c for c, _ in dense}
        by_id.update({c.id: c for c, _ in sparse})
        fused = reciprocal_rank_fusion([[c.id for c, _ in dense], [c.id for c, _ in sparse]])
        ordered = sorted(fused, key=fused.get, reverse=True)[: self.candidates]
        if not ordered:
            return [], None

        texts = [rerank_text(by_id[i]) for i in ordered]
        relevance = max(self.gate.score(query, texts)) if self.gate is not None else None

        if self.reranker is None:
            results = [RetrievedChunk(by_id[i], score=fused[i], rrf_score=fused[i]) for i in ordered]
        else:
            scores = self.reranker.score(query, texts)
            results = [
                RetrievedChunk(by_id[i], score=s, rrf_score=fused[i], rerank_score=s)
                for i, s in zip(ordered, scores)
            ]
            results.sort(key=lambda r: r.score, reverse=True)
            if relevance is None:
                relevance = results[0].score
        return results[: self.top_n], relevance
