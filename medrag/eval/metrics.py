import math
from typing import Iterable


def _unique(items: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def recall_at_k(pages: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(pages[:k]) & gold else 0.0


def reciprocal_rank(pages: list[str], gold: set[str]) -> float:
    for rank, page in enumerate(_unique(pages), start=1):
        if page in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pages: list[str], gold: set[str], k: int) -> float:
    ranking = _unique(pages[:k])
    dcg = sum(1.0 / math.log2(r + 1) for r, p in enumerate(ranking, start=1) if p in gold)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def score_retriever(retriever, items, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    rows = []
    for item in items:
        gold = set(item.gold_pages)
        if not gold:
            continue
        pages = [rc.chunk.page_label for rc in retriever.retrieve(item.question)]
        row = {f"recall@{k}": recall_at_k(pages, gold, k) for k in ks}
        row["mrr"] = reciprocal_rank(pages, gold)
        row["ndcg@5"] = ndcg_at_k(pages, gold, 5)
        rows.append(row)
    keys = [f"recall@{k}" for k in ks] + ["mrr", "ndcg@5"]
    return {**{k: mean(r[k] for r in rows) for k in keys}, "n": len(rows)}
