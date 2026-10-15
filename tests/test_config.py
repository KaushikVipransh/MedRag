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


def test_retrieval_model_defaults():
    s = Settings()
    assert s.rerank_model == "BAAI/bge-reranker-base"
    assert s.gate_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.rerank_threshold == 0.045


def test_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MEDRAG_GATE_MODEL", "")
    assert Settings.from_env(load_dotenv_file=False).gate_model == ""


from medrag.config import CloudSettings

CLOUD_VARS = ("MEDRAG_BACKEND", "SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY", "MEDRAG_FERNET_KEY")


def _clear_cloud(monkeypatch):
    for name in CLOUD_VARS:
        monkeypatch.delenv(name, raising=False)


def test_cloud_defaults_to_local_without_supabase(monkeypatch):
    _clear_cloud(monkeypatch)
    c = CloudSettings.from_env(load_dotenv_file=False)
    assert c.backend == "local"
    assert (c.shared_daily_cap, c.max_books_per_user, c.max_pdf_mb, c.max_pages) == (10, 3, 50, 1000)
    assert (c.global_book_cap, c.storage_cap_mb, c.db_cap_mb, c.cached_books) == (60, 900, 400, 4)


def test_cloud_picks_supabase_when_url_is_set(monkeypatch):
    _clear_cloud(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    c = CloudSettings.from_env(load_dotenv_file=False)
    assert c.backend == "supabase"
    assert c.missing() == ["SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY", "MEDRAG_FERNET_KEY"]


def test_explicit_backend_wins(monkeypatch):
    _clear_cloud(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("MEDRAG_BACKEND", "memory")
    c = CloudSettings.from_env(load_dotenv_file=False)
    assert c.backend == "memory" and c.missing() == []


def test_cloud_limits_from_env(monkeypatch):
    _clear_cloud(monkeypatch)
    monkeypatch.setenv("MEDRAG_SHARED_DAILY_CAP", "25")
    monkeypatch.setenv("MEDRAG_GLOBAL_BOOK_CAP", "5")
    c = CloudSettings.from_env(load_dotenv_file=False)
    assert c.shared_daily_cap == 25 and c.global_book_cap == 5
