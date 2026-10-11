import re
import unicodedata

from rank_bm25 import BM25Okapi

from medrag.types import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in is it its of on or "
    "that the their this to was were what when where which who why with".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return [t for t in _TOKEN.findall(folded) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(
        self, query: str, n: int, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[tuple[Chunk, float]] = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= n:
                break
            if doc_ids and self.chunks[i].doc_id not in doc_ids:
                continue
            hits.append((self.chunks[i], float(scores[i])))
        return hits
