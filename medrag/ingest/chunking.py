import re
from typing import Callable

_UNIT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def _split_long(unit: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    pieces, cur = [], []
    for word in unit.split():
        if cur and count_tokens(" ".join(cur + [word])) > max_tokens:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(word)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 200,
    overlap_tokens: int = 40,
) -> list[str]:
    units: list[str] = []
    for unit in (u.strip() for u in _UNIT_SPLIT.split(text)):
        if not unit:
            continue
        if count_tokens(unit) <= max_tokens:
            units.append(unit)
        else:
            units.extend(_split_long(unit, count_tokens, max_tokens))

    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for unit in units:
        tokens = count_tokens(unit)
        if cur and cur_tokens + tokens > max_tokens:
            chunks.append(" ".join(cur))
            carry: list[str] = []
            carry_tokens = 0
            for prev in reversed(cur):
                t = count_tokens(prev)
                if carry_tokens + t > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_tokens += t
            if carry_tokens + tokens > max_tokens:
                carry, carry_tokens = [], 0
            cur, cur_tokens = carry, carry_tokens
        cur.append(unit)
        cur_tokens += tokens
    if cur:
        chunks.append(" ".join(cur))
    return chunks
