import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from medrag.eval.gold import GoldItem
from medrag.eval.judges import judge_correctness, judge_handling, judge_support, support_scores
from medrag.types import RetrievedChunk


@dataclass
class ItemResult:
    id: str
    category: str
    expected: str
    system: str
    question: str
    answer: str = ""
    refused: bool = False
    refusal_reason: str | None = None
    retrieved_pages: list[str] = field(default_factory=list)
    cited_pages: list[str] = field(default_factory=list)
    top_rerank: float | None = None
    latency_s: float = 0.0
    correctness: str | None = None
    book_support: float | None = None
    citation_support: float | None = None
    handled_ok: bool | None = None
    judge_errors: list[str] = field(default_factory=list)


def _unique(xs):
    return list(dict.fromkeys(xs))


def evaluate_item(
    item: GoldItem,
    system: str,
    answerer,
    judge_llm,
    passages_for: Callable[[str], list[RetrievedChunk]],
) -> ItemResult:
    started = time.perf_counter()
    ans = answerer.answer(item.question)
    r = ItemResult(
        id=item.id, category=item.category, expected=item.expected, system=system, question=item.question,
        answer=ans.text, refused=ans.refused, refusal_reason=ans.refusal_reason,
        retrieved_pages=[rc.chunk.page_label for rc in ans.chunks],
        cited_pages=_unique(c.chunk.page_label for c in ans.citations),
        top_rerank=ans.chunks[0].rerank_score if ans.chunks else None,
        latency_s=round(time.perf_counter() - started, 3),
    )
    if ans.refusal_reason == "llm_error":
        r.judge_errors.append("system llm_error")
        return r

    if item.category == "out_of_scope":
        r.handled_ok = ans.refused
    elif item.category in ("near_miss", "adversarial"):
        if ans.refused:
            r.handled_ok = True
        else:
            j = judge_handling(judge_llm, item.question, ans.text, ans.chunks or passages_for(item.question))
            if j.ok:
                r.handled_ok = j.data["appropriate"]
            else:
                r.judge_errors.append(f"handling: {j.error}")
    elif not ans.refused:
        c = judge_correctness(judge_llm, item.question, item.answer_key, ans.text)
        if c.ok:
            r.correctness = c.data["verdict"]
        else:
            r.judge_errors.append(f"correctness: {c.error}")
        s = judge_support(judge_llm, ans.text, ans.chunks or passages_for(item.question))
        if s.ok:
            r.book_support, r.citation_support = support_scores(s.data["claims"])
        else:
            r.judge_errors.append(f"support: {s.error}")
    return r


def load_results(path: Path) -> list[ItemResult]:
    path = Path(path)
    if not path.exists():
        return []
    return [ItemResult(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_system(items, system, answerer, judge_llm, passages_for, out_path: Path, progress=print) -> list[ItemResult]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r.id: r for r in load_results(out_path)}
    with out_path.open("a", encoding="utf-8") as f:
        for n, item in enumerate(items, 1):
            if item.id in done:
                continue
            r = evaluate_item(item, system, answerer, judge_llm, passages_for)
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            f.flush()
            done[item.id] = r
            progress(f"[{system}] {n}/{len(items)} {item.id} refused={r.refused} correctness={r.correctness} handled={r.handled_ok}")
    return [done[i.id] for i in items if i.id in done]
