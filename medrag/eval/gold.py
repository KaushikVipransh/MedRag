import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

CATEGORIES = ("in_scope", "out_of_scope", "near_miss", "adversarial")
EXPECTED = ("answer", "refuse", "partial_or_refuse")


@dataclass
class GoldItem:
    id: str
    category: str
    question: str
    expected: str
    answer_key: str = ""
    gold_pages: list[str] = field(default_factory=list)
    chapter: str = ""
    verified: bool = False
    notes: str = ""

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")
        if self.expected not in EXPECTED:
            raise ValueError(f"unknown expected behaviour {self.expected!r}")


def _read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_gold(path: Path) -> list[GoldItem]:
    return [GoldItem(**row) for row in _read_jsonl(path)]


def save_gold(path: Path, items: list[GoldItem]) -> None:
    _write_jsonl(path, [asdict(i) for i in items])


def load_evidence(path: Path) -> dict[str, dict]:
    return {row["id"]: {k: v for k, v in row.items() if k != "id"} for row in _read_jsonl(path)}


def save_evidence(path: Path, evidence: dict[str, dict]) -> None:
    _write_jsonl(path, [{"id": k, **v} for k, v in evidence.items()])


def parse_pages(text: str, known: set[str]) -> tuple[list[str], list[str]]:
    """Split a comma-separated page list; return (pages, pages not among `known`).
    With no known labels (index unavailable) nothing is flagged."""
    pages = [p.strip() for p in text.split(",") if p.strip()]
    unknown = [p for p in pages if known and p not in known]
    return pages, unknown


def select(
    items: list[GoldItem],
    include_unverified: bool,
    categories: set[str] | None = None,
    limit_per_category: int | None = None,
) -> list[GoldItem]:
    taken: dict[str, int] = defaultdict(int)
    chosen = []
    for i in items:
        if not (include_unverified or i.verified):
            continue
        if categories is not None and i.category not in categories:
            continue
        if limit_per_category is not None and taken[i.category] >= limit_per_category:
            continue
        taken[i.category] += 1
        chosen.append(i)
    return chosen
