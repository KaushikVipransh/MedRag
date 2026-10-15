import pytest

from medrag.cloud.memory import memory_backend
from medrag.config import CloudSettings
from medrag.library import INTERRUPTED_MESSAGE, Library, UploadRejected, clean_title
from tests.fakes import make_chunk
from tests.fakes_cloud import BODIES, make_pdf_bytes

PDF = make_pdf_bytes(BODIES)


@pytest.fixture
def backend():
    return memory_backend()


def lib(backend, **limits):
    return Library(backend, CloudSettings(backend="memory", **limits))


def test_add_stores_pdf_and_creates_processing_book(backend):
    library = lib(backend)
    book = library.add("alice", "ENT Notes", "ent.pdf", PDF)
    assert (book.status, book.progress, book.pages, book.size_bytes) == ("processing", "Queued", 4, len(PDF))
    assert book.storage_path == f"alice/{book.id}.pdf"
    assert backend.storage.download(book.storage_path) == PDF
    assert library.list("alice") == [book] and library.list("bob") == []


def test_same_file_twice_makes_two_books(backend):
    library = lib(backend)
    a = library.add("alice", "", "ent.pdf", PDF)
    b = library.add("alice", "", "ent.pdf", PDF)
    assert a.id != b.id and len(library.list("alice")) == 2


@pytest.mark.parametrize("title, filename, expected", [
    ("  Dhingra   ENT ", "x.pdf", "Dhingra ENT"),
    ("", "My_Book (2).PDF", "My Book (2)"),
    ("   ", "otología-básica.pdf", "otología básica"),
    ("", ".pdf", "Untitled book"),
    ("T" * 300, "x.pdf", "T" * 200),
])
def test_clean_title(title, filename, expected):
    assert clean_title(title, filename) == expected


def test_per_user_book_limit(backend):
    library = lib(backend, max_books_per_user=1)
    library.add("alice", "", "a.pdf", PDF)
    with pytest.raises(UploadRejected, match=r"up to 1 books \(you have 1\)"):
        library.add("alice", "", "b.pdf", PDF)
    library.add("bob", "", "b.pdf", PDF)  # other users unaffected


def test_global_caps(backend):
    library = lib(backend, global_book_cap=1)
    library.add("alice", "", "a.pdf", PDF)
    with pytest.raises(UploadRejected, match="limit of 1 books"):
        library.add("bob", "", "b.pdf", PDF)


def test_storage_cap(backend):
    library = lib(backend, storage_cap_mb=0)
    with pytest.raises(UploadRejected, match="storage is full"):
        library.add("alice", "", "a.pdf", PDF)


def test_database_cap(backend):
    from medrag.cloud.base import Book

    backend.db.insert_book(Book(id="seed-book", user_id="seed", title="s", filename="s.pdf",
                                storage_path="seed/s.pdf", size_bytes=1))
    backend.db.insert_chunks([{"id": "c", "book_id": "seed-book", "user_id": "seed", "text": "x" * 10,
                               "page_index": 0, "page_label": "1", "section": "", "topic": "", "embedding": [0.0]}])
    with pytest.raises(UploadRejected, match="database is nearly full"):
        lib(backend, db_cap_mb=0).add("alice", "", "a.pdf", PDF)


def test_bad_pdf_is_rejected_and_nothing_is_stored(backend):
    library = lib(backend)
    with pytest.raises(UploadRejected, match="couldn't be opened"):
        library.add("alice", "", "x.pdf", b"not a pdf")
    assert library.list("alice") == []


def test_status_changes(backend):
    library = lib(backend)
    book = library.add("alice", "", "a.pdf", PDF)
    library.set_progress("alice", book.id, "Indexing 10/40")
    assert library.get("alice", book.id).progress == "Indexing 10/40"
    chunk = make_chunk(0, "Tinnitus is ringing.", doc=book.id)
    library.insert_chunks("alice", [chunk], [[1.0, 0.0]])
    library.mark_ready("alice", book.id, pages=4, chunks=1)
    ready = library.get("alice", book.id)
    assert (ready.status, ready.chunks, ready.progress) == ("ready", 1, "")
    library.mark_failed("alice", book.id, "boom")
    failed = library.get("alice", book.id)
    assert (failed.status, failed.error) == ("failed", "boom")
    assert backend.db.load_chunks("alice", book.id) == []


def test_delete_removes_everything_and_notifies(backend):
    library = lib(backend)
    seen = []
    library.on_delete.append(seen.append)
    book = library.add("alice", "", "a.pdf", PDF)
    assert library.delete("bob", book.id) is False and seen == []
    assert library.delete("alice", book.id) is True
    assert library.get("alice", book.id) is None and seen == [book.id]
    with pytest.raises(Exception):
        backend.storage.download(book.storage_path)


def test_fail_interrupted(backend):
    library = lib(backend)
    book = library.add("alice", "", "a.pdf", PDF)
    assert library.fail_interrupted() == 1
    assert library.get("alice", book.id).error == INTERRUPTED_MESSAGE
