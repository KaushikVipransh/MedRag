"""Indexes uploaded books in a single background thread, one at a time, so ingestion never
competes with itself for the Space's two CPUs. A job never raises: any failure marks the book
failed with a readable reason and deletes partial chunks."""
import logging
import queue
import threading
import time

from medrag.cloud.base import Storage
from medrag.config import Settings
from medrag.embeddings import Embedder
from medrag.ingest.build import chunks_for_pdf
from medrag.library import Library
from medrag.types import context_text

log = logging.getLogger(__name__)

NO_TEXT_MESSAGE = "No readable text was found in this PDF."
UNEXPECTED_MESSAGE = "Indexing failed unexpectedly. Delete the book and try uploading it again."


class IngestFailed(Exception):
    """A failure whose message is safe to show to the user."""


class IngestWorker:
    def __init__(
        self, library: Library, storage: Storage, embedder: Embedder, settings: Settings,
        threaded: bool = True, batch: int = 128,
    ):
        self.library = library
        self.storage = storage
        self.embedder = embedder
        self.settings = settings
        self.threaded = threaded
        self.batch = batch
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, user_id: str, book_id: str) -> None:
        if not self.threaded:
            self.run_one(user_id, book_id)
            return
        self._queue.put((user_id, book_id))
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="medrag-ingest", daemon=True)
                self._thread.start()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def _loop(self) -> None:
        while True:
            user_id, book_id = self._queue.get()
            try:
                self.run_one(user_id, book_id)
            except Exception:  # run_one already guards; this keeps the thread alive regardless
                log.exception("ingest job crashed for book %s", book_id)
            finally:
                self._queue.task_done()

    def run_one(self, user_id: str, book_id: str) -> None:
        try:
            book = self.library.get(user_id, book_id)
            if book is None or book.status != "processing":
                return
            self._index(user_id, book)
        except Exception as exc:
            reason = str(exc) if isinstance(exc, IngestFailed) else UNEXPECTED_MESSAGE
            log.exception("ingest failed for book %s", book_id)
            try:
                self.library.mark_failed(user_id, book_id, reason)
            except Exception:
                log.exception("could not mark book %s failed", book_id)

    def _index(self, user_id: str, book) -> None:
        s = self.settings
        self.library.set_progress(user_id, book.id, "Reading the PDF")
        data = self.storage.download(book.storage_path)
        chunks, stats = chunks_for_pdf(
            data, book.id, book.title, self.embedder.count_tokens, s.chunk_max_tokens, s.chunk_overlap_tokens
        )
        if not chunks:
            raise IngestFailed(NO_TEXT_MESSAGE)
        texts = [context_text(c) for c in chunks]
        for start in range(0, len(chunks), self.batch):
            if not self._still_processing(user_id, book.id):
                self.library.discard_chunks(user_id, book.id)
                return
            self.library.set_progress(user_id, book.id, f"Indexing passages {start}/{len(chunks)}")
            part = slice(start, start + self.batch)
            self.library.insert_chunks(user_id, chunks[part], self.embedder.encode(texts[part]))
        if not self.library.mark_ready(user_id, book.id, pages=stats.pages_total, chunks=len(chunks)):
            self.library.discard_chunks(user_id, book.id)  # failed or deleted by someone else meanwhile

    def _still_processing(self, user_id: str, book_id: str) -> bool:
        current = self.library.get(user_id, book_id)
        return current is not None and current.status == "processing"
