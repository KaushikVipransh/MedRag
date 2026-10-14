import json
import random
from collections import Counter, defaultdict

from medrag.eval.gold import GoldItem
from medrag.generation.llm import LLMClient, LLMError
from medrag.types import Chunk

EXCLUDED_SECTIONS = {
    "Cover", "Title Page", "Copyright Page", "Dedication", "Preface",
    "Acknowledgements", "Contents", "Index", "Back Cover",
}

DRAFT_SYSTEM = """You write evaluation questions for a medical textbook question-answering system.
Given one passage from the textbook, write ONE question that a medical student could ask and that the passage answers on its own.

Rules:
- Paraphrase: do not copy distinctive phrases from the passage into the question.
- The question must make sense without seeing the passage (never "according to this passage" or "the figure").
- answer_key: 1-3 sentences giving the correct answer as stated in the passage.
- evidence_quote: one or two consecutive sentences copied EXACTLY, character for character, from the passage, that support the answer.
- If the passage is a figure caption, a reference list, a table fragment without context, or otherwise unsuitable, output {"skip": true}.

Output JSON only: {"question": "...", "answer_key": "...", "evidence_quote": "..."}"""


def parse_json_object(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def quote_in_passage(quote: str, passage: str, min_chars: int = 20) -> bool:
    q = _squash(quote)
    return len(q) >= min_chars and q in _squash(passage)


def sample_chunks(chunks: list[Chunk], n: int, seed: int = 13, min_chars: int = 400) -> list[Chunk]:
    """Round-robin across chapters so every part of the book is represented."""
    rng = random.Random(seed)
    by_chapter: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.section.split(" > ")[0] in EXCLUDED_SECTIONS or len(c.text) < min_chars:
            continue
        by_chapter[c.section].append(c)
    chapters = sorted(by_chapter)
    for ch in chapters:
        rng.shuffle(by_chapter[ch])
    rng.shuffle(chapters)
    picked: list[Chunk] = []
    while len(picked) < n and any(by_chapter[ch] for ch in chapters):
        for ch in chapters:
            if by_chapter[ch] and len(picked) < n:
                picked.append(by_chapter[ch].pop())
    return picked


def draft_item(llm: LLMClient, chunk: Chunk) -> tuple[dict | None, str]:
    messages = [
        {"role": "system", "content": DRAFT_SYSTEM},
        {"role": "user", "content": f"Passage (page {chunk.page_label}, {chunk.section}):\n{chunk.text}"},
    ]
    try:
        raw = llm.complete(messages, temperature=0.0, max_tokens=800)
    except LLMError:
        return None, "llm_error"
    obj = parse_json_object(raw)
    if obj is None:
        return None, "bad_json"
    if obj.get("skip"):
        return None, "skip"
    question, key, quote = (str(obj.get(k, "")).strip() for k in ("question", "answer_key", "evidence_quote"))
    if not question or not key:
        return None, "bad_json"
    if not quote_in_passage(quote, chunk.text):
        return None, "bad_quote"
    return {"question": question, "answer_key": key, "evidence_quote": quote}, "ok"


def draft_gold(
    llm: LLMClient, chunks: list[Chunk], n_target: int, seed: int = 13, oversample: float = 1.5
) -> tuple[list[GoldItem], dict[str, dict], Counter]:
    items: list[GoldItem] = []
    evidence: dict[str, dict] = {}
    stats: Counter = Counter()
    for chunk in sample_chunks(chunks, int(n_target * oversample), seed):
        if len(items) >= n_target:
            break
        draft, status = draft_item(llm, chunk)
        stats[status] += 1
        if draft is None:
            continue
        item_id = f"in-{len(items) + 1:03d}"
        items.append(GoldItem(
            id=item_id, category="in_scope", question=draft["question"], expected="answer",
            answer_key=draft["answer_key"], gold_pages=[chunk.page_label], chapter=chunk.section,
        ))
        evidence[item_id] = {"chunk_id": chunk.id, "quote": draft["evidence_quote"]}
    return items, evidence, stats
