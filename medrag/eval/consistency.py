import math
from itertools import combinations

from medrag.eval.metrics import mean
from medrag.types import Answer


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cosine(u: list[float], v: list[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v))
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


def consistency_scores(runs: list[list[Answer]], embed) -> dict:
    """`runs[q]` holds repeated answers to question q. Text similarity uses the
    embedder; citation stability uses the Jaccard overlap of cited page sets."""
    per_question = []
    for answers in runs:
        vectors = embed([a.text for a in answers])
        pages = [{c.chunk.page_label for c in a.citations} for a in answers]
        pairs = list(combinations(range(len(answers)), 2))
        per_question.append({
            "text_cosine": mean(cosine(vectors[i], vectors[j]) for i, j in pairs),
            "citation_jaccard": mean(jaccard(pages[i], pages[j]) for i, j in pairs),
        })
    return {
        "text_cosine": mean(q["text_cosine"] for q in per_question),
        "citation_jaccard": mean(q["citation_jaccard"] for q in per_question),
        "per_question": per_question,
    }
