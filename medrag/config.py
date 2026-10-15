import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    temperature: float = 0.1
    chroma_dir: Path = Path("data/chroma")
    collection: str = "medrag"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "BAAI/bge-reranker-base"  # orders passages
    gate_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # decides answerability; "" disables
    chunk_max_tokens: int = 200
    chunk_overlap_tokens: int = 40
    candidates_per_retriever: int = 30
    top_n: int = 5
    rerank_threshold: float = 0.045  # minimum gate relevance (sigmoid): weakest verified in-scope 0.057, strongest out-of-scope 0.037
    max_question_chars: int = 1000

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "Settings":
        if load_dotenv_file:
            from dotenv import load_dotenv

            load_dotenv()
        d = cls()
        env = os.getenv
        return cls(
            groq_api_key=env("GROQ_API_KEY", ""),
            llm_model=env("MEDRAG_LLM_MODEL", d.llm_model),
            temperature=float(env("MEDRAG_TEMPERATURE", d.temperature)),
            chroma_dir=Path(env("MEDRAG_CHROMA_DIR", str(d.chroma_dir))),
            collection=env("MEDRAG_COLLECTION", d.collection),
            embed_model=env("MEDRAG_EMBED_MODEL", d.embed_model),
            rerank_model=env("MEDRAG_RERANK_MODEL", d.rerank_model),
            gate_model=env("MEDRAG_GATE_MODEL", d.gate_model),
            chunk_max_tokens=int(env("MEDRAG_CHUNK_MAX_TOKENS", d.chunk_max_tokens)),
            chunk_overlap_tokens=int(env("MEDRAG_CHUNK_OVERLAP_TOKENS", d.chunk_overlap_tokens)),
            candidates_per_retriever=int(env("MEDRAG_CANDIDATES", d.candidates_per_retriever)),
            top_n=int(env("MEDRAG_TOP_N", d.top_n)),
            rerank_threshold=float(env("MEDRAG_RERANK_THRESHOLD", d.rerank_threshold)),
            max_question_chars=int(env("MEDRAG_MAX_QUESTION_CHARS", d.max_question_chars)),
        )


@dataclass(frozen=True)
class CloudSettings:
    """Multi-user settings. backend: "local" (single-book Chroma app), "memory" (multi-user,
    nothing persisted; for development and tests) or "supabase" (production)."""

    backend: str = "local"
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""
    fernet_key: str = ""
    shared_daily_cap: int = 10
    max_books_per_user: int = 3
    max_pdf_mb: int = 50
    max_pages: int = 1000
    global_book_cap: int = 60
    storage_cap_mb: int = 900
    db_cap_mb: int = 400
    cached_books: int = 4

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "CloudSettings":
        if load_dotenv_file:
            from dotenv import load_dotenv

            load_dotenv()
        d = cls()
        env = os.getenv
        url = env("SUPABASE_URL", "")
        return cls(
            backend=env("MEDRAG_BACKEND", "") or ("supabase" if url else "local"),
            supabase_url=url,
            supabase_anon_key=env("SUPABASE_ANON_KEY", ""),
            supabase_service_key=env("SUPABASE_SERVICE_KEY", ""),
            fernet_key=env("MEDRAG_FERNET_KEY", ""),
            shared_daily_cap=int(env("MEDRAG_SHARED_DAILY_CAP", d.shared_daily_cap)),
            max_books_per_user=int(env("MEDRAG_MAX_BOOKS_PER_USER", d.max_books_per_user)),
            max_pdf_mb=int(env("MEDRAG_MAX_PDF_MB", d.max_pdf_mb)),
            max_pages=int(env("MEDRAG_MAX_PAGES", d.max_pages)),
            global_book_cap=int(env("MEDRAG_GLOBAL_BOOK_CAP", d.global_book_cap)),
            storage_cap_mb=int(env("MEDRAG_STORAGE_CAP_MB", d.storage_cap_mb)),
            db_cap_mb=int(env("MEDRAG_DB_CAP_MB", d.db_cap_mb)),
            cached_books=int(env("MEDRAG_CACHED_BOOKS", d.cached_books)),
        )

    def missing(self) -> list[str]:
        """Names of required settings that are empty for the chosen backend."""
        if self.backend != "supabase":
            return []
        required = {
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_ANON_KEY": self.supabase_anon_key,
            "SUPABASE_SERVICE_KEY": self.supabase_service_key,
            "MEDRAG_FERNET_KEY": self.fernet_key,
        }
        return [name for name, value in required.items() if not value]
