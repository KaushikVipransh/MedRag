import hashlib


def split_dev_test(items):
    dev, test = [], []
    for item in items:
        bucket = int(hashlib.md5(item.id.encode()).hexdigest(), 16) % 2
        (dev if bucket == 0 else test).append(item)
    return dev, test


def rates(scores_in: list[float], scores_out: list[float], t: float) -> tuple[float | None, float | None]:
    passed = sum(s >= t for s in scores_in) / len(scores_in) if scores_in else None
    rejected = sum(s < t for s in scores_out) / len(scores_out) if scores_out else None
    return passed, rejected


def choose_threshold(scores_in: list[float], scores_out: list[float], min_pass: float = 0.98) -> float:
    """Largest threshold that still lets `min_pass` of in-scope questions through, which
    maximises rejection of out-of-scope questions."""
    best = 0.0
    for t in sorted({0.0, *scores_in, *scores_out}):
        passed, _ = rates(scores_in, scores_out, t)
        if passed is not None and passed >= min_pass:
            best = t
    return best
