import pytest

from medrag.cloud.memory import memory_backend
from medrag.config import CloudSettings, Settings
from medrag.ingest.job import UNEXPECTED_MESSAGE, IngestWorker
from medrag.library import Library
from tests.fakes import FakeEmbedder
from tests.fakes_cloud import BODIES, make_pdf_bytes

SETTINGS = Settings(chunk_max_tokens=50, chunk_overlap_tokens=10)


@pytest.fixture
def parts():
    backend = memory_backend()
    return backend, Library(backend, CloudSettings(backend="memory"))


def test_ingests_a_book_to_ready(parts):
    backend, library = parts
    book = library.add("alice", "ENT", "ent.pdf", make_pdf_bytes(BODIES))
    IngestWorker(library, backend.storage, FakeEmbedder(), SETTINGS, threaded=False, batch=2).submit("alice", book.id)
    done = library.get("alice", book.id)
    assert done.status == "ready" and done.chunks > 0 and done.pages == 4
    rows = backend.db.load_chunks("alice", book.id)
    assert len(rows) == done.chunks
    assert all(r["user_id"] == "alice" and r["book_id"] == book.id and len(r["embedding"]) == 64 for r in rows)
    assert rows[0]["page_label"] == "101"


def test_progress_is_reported(parts):
    backend, library = parts
    book = library.add("alice", "", "ent.pdf", make_pdf_bytes(BODIES))
    seen = []
    original = library.set_progress
    library.set_progress = lambda u, b, text: (seen.append(text), original(u, b, text))
    IngestWorker(library, backend.storage, FakeEmbedder(), SETTINGS, threaded=False, batch=2).run_one("alice", book.id)
    assert seen[0] == "Reading the PDF" and any(t.startswith("Indexing passages") for t in seen)


def test_missing_pdf_fails_cleanly(parts):
    backend, library = parts
    book = library.add("alice", "", "ent.pdf", make_pdf_bytes(BODIES))
    backend.storage.remove(book.storage_path)
    IngestWorker(library, backend.storage, FakeEmbedder(), SETTINGS, threaded=False).run_one("alice", book.id)
    failed = library.get("alice", book.id)
    assert failed.status == "failed" and failed.error == UNEXPECTED_MESSAGE


def test_ignores_books_that_are_not_processing(parts):
    backend, library = parts
    book = library.add("alice", "", "ent.pdf", make_pdf_bytes(BODIES))
    library.mark_failed("alice", book.id, "earlier")
    IngestWorker(library, backend.storage, FakeEmbedder(), SETTINGS, threaded=False).run_one("alice", book.id)
    assert library.get("alice", book.id).error == "earlier"


class DeletingEmbedder(FakeEmbedder):
    """Deletes the book the first time it is asked to embed, as a user would mid-ingest."""

    def __init__(self, library, user_id, book_id):
        self.library, self.user_id, self.book_id, self.fired = library, user_id, book_id, False

    def encode(self, texts):
        if not self.fired:
            self.fired = True
            self.library.delete(self.user_id, self.book_id)
        return super().encode(texts)


def test_deleting_during_ingestion_leaves_no_chunks_and_the_queue_keeps_going(parts):
    backend, library = parts
    doomed = library.add("alice", "", "a.pdf", make_pdf_bytes(BODIES))
    later = library.add("alice", "", "b.pdf", make_pdf_bytes(BODIES))
    worker = IngestWorker(library, backend.storage, DeletingEmbedder(library, "alice", doomed.id), SETTINGS, batch=2)
    worker.submit("alice", doomed.id)
    worker.submit("alice", later.id)
    assert worker.wait_idle(timeout=30)
    assert library.get("alice", doomed.id) is None
    assert backend.db.load_chunks("alice", doomed.id) == []
    assert library.get("alice", later.id).status == "ready"


def test_a_crashing_job_does_not_kill_the_worker_thread(parts, monkeypatch):
    backend, library = parts
    first = library.add("alice", "", "a.pdf", make_pdf_bytes(BODIES))
    second = library.add("alice", "", "b.pdf", make_pdf_bytes(BODIES))
    real_get = library.get

    def exploding_get(user_id, book_id):
        if book_id == first.id:
            raise RuntimeError("database fell over")
        return real_get(user_id, book_id)

    monkeypatch.setattr(library, "get", exploding_get)
    worker = IngestWorker(library, backend.storage, FakeEmbedder(), SETTINGS)
    worker.submit("alice", first.id)
    worker.submit("alice", second.id)
    assert worker.wait_idle(timeout=30)
    assert backend.db.get_book("alice", first.id).status == "failed"
    assert backend.db.get_book("alice", second.id).status == "ready"
