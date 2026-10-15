"""Builds the multi-user app's parts once per process and offers the two actions that span
several parts (upload, delete)."""
from dataclasses import dataclass
from typing import Callable

from cryptography.fernet import Fernet

from medrag.accounts import Accounts
from medrag.chat import ChatService
from medrag.cloud.base import Backend, Book
from medrag.config import CloudSettings, Settings
from medrag.embeddings import Embedder
from medrag.generation.llm import LLMClient
from medrag.ingest.job import IngestWorker
from medrag.keys import KeyVault, check_groq_key
from medrag.library import Library, UploadRejected
from medrag.quota import Quota
from medrag.retrieval.rerank import Reranker

CONFIRM_MESSAGE = "Please confirm this is a medical or health book and that you have the right to use it."


@dataclass
class Models:
    embedder: Embedder
    reranker: Reranker | None
    gate: Reranker | None
    llm_factory: Callable[[str], LLMClient]


@dataclass
class Services:
    settings: Settings
    cloud: CloudSettings
    backend: Backend
    accounts: Accounts
    library: Library
    vault: KeyVault
    quota: Quota
    worker: IngestWorker
    chat: ChatService


def assemble(
    backend: Backend, models: Models, settings: Settings, cloud: CloudSettings,
    key_checker: Callable[[str], bool] = check_groq_key, threaded: bool = True,
) -> Services:
    library = Library(backend, cloud)
    library.fail_interrupted()
    vault = KeyVault(backend.db, cloud.fernet_key or Fernet.generate_key().decode(), checker=key_checker)
    quota = Quota(backend.db, vault, settings.groq_api_key, cloud.shared_daily_cap)
    chat = ChatService(library, backend.db, models, quota, settings, cache_size=cloud.cached_books)
    library.on_delete.append(chat.evict)
    worker = IngestWorker(library, backend.storage, models.embedder, settings, threaded=threaded)
    return Services(settings, cloud, backend, Accounts(backend.auth), library, vault, quota, worker, chat)


def upload_book(services: Services, user_id: str, title: str, filename: str, data: bytes, confirmed: bool) -> Book:
    if not confirmed:
        raise UploadRejected(CONFIRM_MESSAGE)
    book = services.library.add(user_id, title, filename, data)
    services.worker.submit(user_id, book.id)
    return book


def delete_book(services: Services, user_id: str, book_id: str) -> bool:
    return services.library.delete(user_id, book_id)


def make_backend(cloud: CloudSettings) -> Backend:
    if cloud.backend == "memory":
        from medrag.cloud.memory import memory_backend

        return memory_backend()
    if cloud.backend == "supabase":
        from medrag.cloud.supabase import supabase_backend

        return supabase_backend(cloud.supabase_url, cloud.supabase_service_key, cloud.supabase_anon_key)
    raise ValueError(f"MEDRAG_BACKEND must be 'memory' or 'supabase' for the multi-user app, not {cloud.backend!r}")


def load_models(settings: Settings) -> Models:
    from medrag.embeddings import SentenceTransformerEmbedder
    from medrag.generation.llm import GroqClient
    from medrag.retrieval.rerank import CrossEncoderReranker

    return Models(
        embedder=SentenceTransformerEmbedder(settings.embed_model),
        reranker=CrossEncoderReranker(settings.rerank_model),
        gate=CrossEncoderReranker(settings.gate_model) if settings.gate_model else None,
        llm_factory=lambda api_key: GroqClient(api_key, settings.llm_model),
    )


def build_services(settings: Settings, cloud: CloudSettings) -> Services:
    missing = cloud.missing()
    if missing:
        raise ValueError(f"Missing settings: {', '.join(missing)}")
    return assemble(make_backend(cloud), load_models(settings), settings, cloud)
