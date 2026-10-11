import re

from medrag.generation.prompts import REFUSAL_TOKEN

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Reasoning models (e.g. gpt-oss) often cite as 【1】 or 【1†L1-L3】 instead of [1].
_ALT_MARKER = re.compile(r"[【\[]\s*(\d+)\s*(?:†[^】\]]*)?[】\]]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\]])\s+(?!\[)|\n+")


def normalize_markers(text: str) -> str:
    return _ALT_MARKER.sub(lambda m: f"[{m.group(1)}]", text)


def _numbers(group: str) -> list[int]:
    return [int(x) for x in re.split(r"\s*,\s*", group)]


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    invalid: list[int] = []
    for m in _MARKER.finditer(text):
        for n in _numbers(m.group(1)):
            bucket = valid if 1 <= n <= n_sources else invalid
            if n not in bucket:
                bucket.append(n)
    return valid, invalid


def remove_invalid_markers(text: str, n_sources: int) -> str:
    def repl(m: re.Match) -> str:
        keep = [str(n) for n in _numbers(m.group(1)) if 1 <= n <= n_sources]
        return f"[{', '.join(keep)}]" if keep else ""

    return re.sub(r"\s+\.", ".", _MARKER.sub(repl, text)) if _MARKER.search(text) else text


def uncited_sentences(text: str, min_chars: int = 40) -> list[str]:
    out = []
    for sentence in _SENTENCE_SPLIT.split(text):
        s = sentence.strip()
        if len(s) < min_chars or s.endswith(":"):
            continue
        if not _MARKER.search(s):
            out.append(s)
    return out


def is_refusal(text: str) -> bool:
    return text.strip().upper().startswith(REFUSAL_TOKEN)
