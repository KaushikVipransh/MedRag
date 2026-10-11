from pathlib import Path

from medrag.eval.config import EvalSettings


def test_defaults():
    s = EvalSettings()
    assert s.judge_model == "qwen/qwen3.8-27b" and s.drafter_model == "qwen/qwen3.8-27b"
    assert s.gold_path == Path("data/eval/gold.jsonl")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MEDRAG_GOLD_PATH", "/tmp/g.jsonl")
    monkeypatch.setenv("MEDRAG_JUDGE_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("MEDRAG_EVAL_RESULTS", "/tmp/res")
    s = EvalSettings.from_env()
    assert s.gold_path == Path("/tmp/g.jsonl") and s.judge_model == "openai/gpt-oss-20b"
    assert s.results_dir == Path("/tmp/res")
