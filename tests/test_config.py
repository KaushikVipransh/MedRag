from pathlib import Path

from medrag.config import Settings


def test_defaults():
    s = Settings()
    assert s.llm_model == "openai/gpt-oss-120b"
    assert s.top_n == 5 and s.chunk_max_tokens == 200 and s.temperature == 0.1


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("MEDRAG_TOP_N", "7")
    monkeypatch.setenv("MEDRAG_RERANK_THRESHOLD", "0.2")
    monkeypatch.setenv("MEDRAG_CHROMA_DIR", "somewhere/db")
    s = Settings.from_env(load_dotenv_file=False)
    assert s.groq_api_key == "k"
    assert s.top_n == 7
    assert s.rerank_threshold == 0.2
    assert s.chroma_dir == Path("somewhere/db")


def test_missing_env_uses_defaults(monkeypatch):
    monkeypatch.delenv("MEDRAG_LLM_MODEL", raising=False)
    assert Settings.from_env(load_dotenv_file=False).llm_model == "openai/gpt-oss-120b"
