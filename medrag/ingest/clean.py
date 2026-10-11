import re
import unicodedata
from collections import Counter

_DIGITS = re.compile(r"\d+")
_BULLET = re.compile(r"^\s*(?:[•\-–▪●◦*]|\(?[0-9ivxlc]+[.)]|\(?[a-z][.)])\s+", re.IGNORECASE)
_ENDS_WITH_PAGE = re.compile(r"\d+\s*$")
_NON_CONTENT_TITLES = {
    "contents", "table of contents", "index", "subject index",
    "bibliography", "references", "further reading",
}


def _norm(line: str) -> str:
    return _DIGITS.sub("#", line.strip().lower())


def _margin(lines: list[str], margin: int) -> list[str]:
    return lines[:margin] + lines[-margin:]


def _normalize_chars(line: str) -> str:
    """Expand ligatures (NFKC) and turn unassigned code points, which some PDFs emit
    after bullet glyphs, into spaces."""
    line = unicodedata.normalize("NFKC", line)
    return "".join(" " if unicodedata.category(ch) == "Cn" else ch for ch in line)


def find_repeating_margin_lines(
    pages: list[list[str]], margin: int = 3, min_count: int = 4
) -> set[str]:
    """Normalised lines (digits -> '#') that sit in the top/bottom `margin` lines of at
    least `min_count` pages: running headers, footers and bare page numbers. Lines that
    also occur in page bodies (recurring sub-headings like "Treatment") are kept as
    content unless they contain no letters."""
    margin_counts: Counter[str] = Counter()
    body_counts: Counter[str] = Counter()
    for lines in pages:
        n = len(lines)
        in_margin: set[str] = set()
        in_body: set[str] = set()
        for i, line in enumerate(lines):
            if not line.strip() or len(line) > 80:
                continue
            (in_margin if i < margin or i >= n - margin else in_body).add(_norm(line))
        margin_counts.update(in_margin)
        body_counts.update(in_body)
    return {
        line
        for line, n in margin_counts.items()
        if n >= min_count and (not any(c.isalpha() for c in line) or body_counts[line] * 4 < n)
    }


def strip_margin_lines(lines: list[str], repeating: set[str], margin: int = 3) -> list[str]:
    n = len(lines)
    return [
        line
        for i, line in enumerate(lines)
        if not ((i < margin or i >= n - margin) and _norm(line) in repeating)
    ]


def lines_to_text(lines: list[str]) -> str:
    """Join PDF lines into paragraphs: rejoin hyphenated words, keep list items and
    blank lines as paragraph breaks."""
    paras: list[str] = []
    cur = ""
    for raw in lines:
        line = _normalize_chars(raw).strip()
        if not line:
            if cur:
                paras.append(cur)
                cur = ""
            continue
        if cur and _BULLET.match(line):
            paras.append(cur)
            cur = line
        elif not cur:
            cur = line
        elif cur.endswith("-"):
            cur = cur[:-1] + line if line[:1].islower() else cur + line
        else:
            cur = f"{cur} {line}"
    if cur:
        paras.append(cur)
    return "\n".join(paras)


def is_non_content_page(lines: list[str], min_lines: int = 15, ratio: float = 0.5) -> bool:
    """Contents, index and bibliography pages: titled as such, or mostly lines ending
    in page numbers."""
    stripped = [l.strip() for l in lines if l.strip()]
    if not stripped:
        return True
    if stripped[0].lower().rstrip(":") in _NON_CONTENT_TITLES:
        return True
    if len(stripped) >= min_lines:
        ending_in_number = sum(bool(_ENDS_WITH_PAGE.search(l)) for l in stripped)
        if ending_in_number / len(stripped) > ratio:
            return True
    return False
