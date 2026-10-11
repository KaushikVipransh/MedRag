import re
from collections import Counter

_MARGIN_NUMBER = re.compile(r"^(\d{1,4})(?:\s|$)|\s(\d{1,4})$")


def _labels_are_meaningful(pdf_labels: list[str] | None, n: int) -> bool:
    if not pdf_labels or not any(pdf_labels):
        return False
    return pdf_labels != [str(i + 1) for i in range(n)]


def infer_page_labels(
    pages: list[list[str]], pdf_labels: list[str] | None = None, margin: int = 3
) -> list[str]:
    """Return the printed page number for every PDF page.

    1. Meaningful embedded PDF page labels are used as-is.
    2. Otherwise numbers at the start/end of margin lines vote for an offset
       (printed - pdf_index); the most common offset wins. Pages before printed
       page 1 get 'front-<pdf page>'.
    3. With no evidence, fall back to the 1-based PDF index.
    """
    n = len(pages)
    if _labels_are_meaningful(pdf_labels, n):
        return [label or str(i + 1) for i, label in enumerate(pdf_labels)]

    votes: Counter[int] = Counter()
    for i, lines in enumerate(pages):
        for line in lines[:margin] + lines[-margin:]:
            m = _MARGIN_NUMBER.search(line.strip())
            if m:
                votes[int(m.group(1) or m.group(2)) - i] += 1
    if not votes:
        return [str(i + 1) for i in range(n)]

    offset = votes.most_common(1)[0][0]
    return [str(i + offset) if i + offset >= 1 else f"front-{i + 1}" for i in range(n)]
