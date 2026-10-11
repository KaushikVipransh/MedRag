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
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_max_tokens: int = 200
    chunk_overlap_tokens: int = 40
    candidates_per_retriever: int = 30
    top_n: int = 5
    rerank_threshold: float = 0.01  # sigmoid relevance; calibrated in Plan 2
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
            chunk_max_tokens=int(env("MEDRAG_CHUNK_MAX_TOKENS", d.chunk_max_tokens)),
            chunk_overlap_tokens=int(env("MEDRAG_CHUNK_OVERLAP_TOKENS", d.chunk_overlap_tokens)),
            candidates_per_retriever=int(env("MEDRAG_CANDIDATES", d.candidates_per_retriever)),
            top_n=int(env("MEDRAG_TOP_N", d.top_n)),
            rerank_threshold=float(env("MEDRAG_RERANK_THRESHOLD", d.rerank_threshold)),
            max_question_chars=int(env("MEDRAG_MAX_QUESTION_CHARS", d.max_question_chars)),
        )
