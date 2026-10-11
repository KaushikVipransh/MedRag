import hashlib
import math
import re

from medrag.types import Chunk

_WORD = re.compile(r"[a-z0-9]+")


def make_chunk(i: int, text: str, doc: str = "book-a", page: int = 0, section: str = "Ear") -> Chunk:
    return Chunk(
        id=f"{doc}-p{page:04d}-c{i:03d}",
        doc_id=doc,
        doc_title=doc.replace("-", " ").title(),
        text=text,
        page_index=page,
        page_label=str(page + 1),
        section=section,
    )


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: similar words -> similar vectors."""

    dim = 64

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [1e-6] * self.dim
            for word in _WORD.findall(text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            out.append([x / norm for x in vec])
        return out

    def count_tokens(self, text: str) -> int:
        return len(text.split())


from medrag.retrieval.bm25 import tokenize  # noqa: E402


class OverlapReranker:
    """Score = fraction of query terms present in the passage."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        terms = set(tokenize(query))
        if not terms:
            return [0.0] * len(texts)
        return [len(terms & set(tokenize(t))) / len(terms) for t in texts]


class FixedReranker:
    """Returns a preset score per passage text substring; 0.0 otherwise."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [next((v for k, v in self.scores.items() if k in t), 0.0) for t in texts]


class FakeLLM:
    def __init__(self, responses: list[str] | None = None, error: Exception | None = None):
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[list[dict]] = []

    def complete(self, messages, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.responses.pop(0)
