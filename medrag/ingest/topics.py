"""Topic headings: the all-caps heading lines a textbook uses to name the topic under
discussion (e.g. "WEGENER'S GRANULOMATOSIS"). Headings that recur across many pages
("TREATMENT", "CLINICAL FEATURES") are generic and are not topics."""
import re
import string
from collections import Counter

_SPACED_LETTERS = re.compile(r"^(\S )+\S$")
_NOT_HEADINGS = ("TABLE", "FIGURE", "SECTION", "CHAPTER")
# Numbered or lettered list entries ("3. X", "A. X", "(a) X"): their siblings often run
# inline in the prose, so treating them as topics would mislabel later passages.
_ENUMERATED = re.compile(r"^\(?[0-9ivxlcA-Za-z]{1,3}[.)]\s")


def _norm(line: str) -> str:
    return " ".join(line.split())


def is_caps_heading(line: str) -> bool:
    s = _norm(line)
    if not s or len(s) > 70 or not s.isupper() or _SPACED_LETTERS.match(s) or s.startswith(_NOT_HEADINGS):
        return False
    if not s[0].isalpha() or _ENUMERATED.match(s):
        return False
    return any(sum(ch.isalpha() for ch in word) >= 4 for word in s.split())


def raw_headings(lines: list[str]) -> list[str]:
    """Caps heading lines in order; a line ending in a space continues onto the next."""
    out: list[str] = []
    pending = ""
    for raw in lines:
        if is_caps_heading(raw) or (pending and _norm(raw).isupper() and _norm(raw)):
            pending = f"{pending} {_norm(raw)}".strip()
            if not raw.endswith(" "):
                out.append(pending)
                pending = ""
        elif pending:
            out.append(pending)
            pending = ""
    if pending:
        out.append(pending)
    return out


def generic_headings(pages: list[list[str]], min_pages: int = 3) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update(set(raw_headings(lines)))
    return {h for h, n in counts.items() if n >= min_pages}


def pretty(heading: str) -> str:
    return string.capwords(heading.lower())


def topic_headings(lines: list[str], generic: set[str]) -> list[str]:
    return [pretty(h) for h in raw_headings(lines) if h not in generic]
