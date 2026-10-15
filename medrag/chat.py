"""Answers a user's question from one of their ready books. Checks ownership, picks the Groq
key (shared under the daily cap, then the user's own), falls back from an exhausted shared key,
and counts only answers the model generated on the shared key."""
import threading
from collections import OrderedDict

from medrag.cloud.base import Book, Database
from medrag.config import Settings
from medrag.generation.generator import Generator
from medrag.library import Library
from medrag.pipeline import MedRAG
from medrag.quota import KeyChoice, Quota, counts_toward_cap
from medrag.retrieval.bm25 import BM25Index
from medrag.retrieval.retriever import HybridRetriever
from medrag.store_pg import InMemoryChunkStore
from medrag.types import Answer

BOOK_UNAVAILABLE_MESSAGE = "This book isn't available. Pick a ready book from your library."
QUOTA_MESSAGE = "You've used today's {cap} free questions. Add your free Groq key in Settings to keep going."
SHARED_EXHAUSTED_MESSAGE = "The free shared quota is used up for today. Add your free Groq key in Settings."
OWN_KEY_REJECTED_MESSAGE = "Your Groq key was rejected. Check or replace it in Settings."
OWN_KEY_LIMIT_MESSAGE = "Your Groq key has hit its daily limit. Try again later or check your Groq account."
_KEY_FAILURES = ("rate_limited", "key_rejected")


class ChatService:
    def __init__(self, library: Library, db: Database, models, quota: Quota, settings: Settings, cache_size: int = 4):
        self.library = library
        self.db = db
        self.models = models
        self.quota = quota
        self.settings = settings
        self.cache_size = cache_size
        self._cache: OrderedDict[str, HybridRetriever] = OrderedDict()
        self._lock = threading.Lock()

    def ask(self, user_id: str, book_id: str, question: str, history: list[dict] | None = None) -> Answer:
        book = self.library.get(user_id, book_id)
        if book is None or book.status != "ready":
            return Answer(text=BOOK_UNAVAILABLE_MESSAGE, refused=True, refusal_reason="book_unavailable")
        choice = self.quota.choose(user_id)
        if choice.source == "none":
            return Answer(text=QUOTA_MESSAGE.format(cap=self.quota.daily_cap), refused=True, refusal_reason="quota")

        retriever = self._retriever(book)
        answer = self._answer(retriever, choice.api_key, question, history, book_id)
        if choice.source == "shared" and answer.refusal_reason in _KEY_FAILURES:
            own = self.quota.own_key(user_id)
            if not own:
                answer.text = SHARED_EXHAUSTED_MESSAGE
                return answer
            choice = KeyChoice("own", own)
            answer = self._answer(retriever, own, question, history, book_id)
        if choice.source == "own" and answer.refusal_reason == "key_rejected":
            answer.text = OWN_KEY_REJECTED_MESSAGE
        elif choice.source == "own" and answer.refusal_reason == "rate_limited":
            answer.text = OWN_KEY_LIMIT_MESSAGE
        if choice.source == "shared" and counts_toward_cap(answer):
            self.quota.record(user_id)
        return answer

    def evict(self, book_id: str) -> None:
        with self._lock:
            self._cache.pop(book_id, None)

    def cached_ids(self) -> list[str]:
        with self._lock:
            return list(self._cache)

    def _answer(self, retriever: HybridRetriever, api_key: str, question: str, history, book_id: str) -> Answer:
        llm = self.models.llm_factory(api_key)
        rag = MedRAG(retriever, Generator(llm, self.settings.temperature), llm, self.settings)
        return rag.answer(question, history=history, doc_ids=[book_id])

    def _retriever(self, book: Book) -> HybridRetriever:
        with self._lock:
            if book.id in self._cache:
                self._cache.move_to_end(book.id)
                return self._cache[book.id]
        store = InMemoryChunkStore.from_rows(self.db.load_chunks(book.user_id, book.id), book.title)
        m = self.models
        retriever = HybridRetriever(
            store=store, embedder=m.embedder, bm25=BM25Index(store.all_chunks()), reranker=m.reranker,
            candidates=self.settings.candidates_per_retriever, top_n=self.settings.top_n, gate=m.gate,
        )
        with self._lock:
            self._cache[book.id] = retriever
            self._cache.move_to_end(book.id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return retriever
