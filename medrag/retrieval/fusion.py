from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Combine ranked id lists by summing 1 / (k + rank). Scale-free, so dense and
    BM25 scores never have to be compared directly."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return dict(scores)
