import pytest

from medrag.library import UploadRejected
from medrag.services import delete_book, upload_book
from tests.fakes_cloud import BODIES, make_pdf_bytes, make_services, ready_book


def test_upload_requires_confirmation():
    s, _ = make_services()
    with pytest.raises(UploadRejected, match="confirm"):
        upload_book(s, "alice", "", "a.pdf", make_pdf_bytes(BODIES), confirmed=False)
    assert s.library.list("alice") == []


def test_upload_indexes_the_book():
    s, _ = make_services()
    assert ready_book(s, "alice").status == "ready"


def test_delete_evicts_cached_retriever():
    s, _ = make_services()
    book = ready_book(s, "alice")
    s.chat.ask("alice", book.id, "What is tinnitus?")
    assert book.id in s.chat.cached_ids()
    assert delete_book(s, "alice", book.id)
    assert book.id not in s.chat.cached_ids()


def test_assemble_fails_interrupted_books():
    from medrag.services import assemble

    s, _ = make_services()
    stuck = s.library.add("alice", "", "a.pdf", make_pdf_bytes(BODIES))  # added but never submitted
    again = assemble(s.backend, s.chat.models, s.settings, s.cloud, key_checker=lambda k: True, threaded=False)
    assert again.library.get("alice", stuck.id).status == "failed"
