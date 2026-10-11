import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalSettings:
    gold_path: Path = Path("data/eval/gold.jsonl")
    evidence_path: Path = Path("data/eval/evidence.jsonl")
    cache_dir: Path = Path("data/eval_cache")
    results_dir: Path = Path("data/eval_results")
    report_dir: Path = Path("docs/results")
    judge_model: str = "qwen/qwen3.8-27b"
    drafter_model: str = "qwen/qwen3.8-27b"

    @classmethod
    def from_env(cls) -> "EvalSettings":
        d = cls()
        env = os.getenv
        return cls(
            gold_path=Path(env("MEDRAG_GOLD_PATH", str(d.gold_path))),
            evidence_path=Path(env("MEDRAG_EVIDENCE_PATH", str(d.evidence_path))),
            cache_dir=Path(env("MEDRAG_EVAL_CACHE", str(d.cache_dir))),
            results_dir=Path(env("MEDRAG_EVAL_RESULTS", str(d.results_dir))),
            report_dir=Path(env("MEDRAG_REPORT_DIR", str(d.report_dir))),
            judge_model=env("MEDRAG_JUDGE_MODEL", d.judge_model),
            drafter_model=env("MEDRAG_DRAFTER_MODEL", d.drafter_model),
        )
