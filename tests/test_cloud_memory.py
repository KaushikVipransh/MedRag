from datetime import date

import pytest

from medrag.cloud.base import AuthFailed, BackendError, Book
from medrag.cloud.memory import memory_backend


def book(book_id="b1", user="alice", status="processing", size=100):
    return Book(id=book_id, user_id=user, title="T", filename="t.pdf",
                storage_path=f"{user}/{book_id}.pdf", size_bytes=size, status=status)


def row(chunk_id, book_id="b1", user="alice"):
    return {"id": chunk_id, "book_id": book_id, "user_id": user, "text": "x", "page_index": 0,
            "page_label": "1", "section": "", "topic": "", "embedding": [1.0, 0.0]}


@pytest.fixture
def db():
    return memory_backend().db


def test_books_are_private_to_their_owner(db):
    db.insert_book(book())
    assert db.get_book("alice", "b1").title == "T"
    assert db.get_book("bob", "b1") is None
    assert db.list_books("bob") == []
    db.update_book("bob", "b1", {"status": "failed"})
    db.delete_book("bob", "b1")
    assert db.get_book("alice", "b1").status == "processing"


def test_chunks_are_private_and_cascade_on_delete(db):
    db.insert_book(book())
    db.insert_chunks([row("c2"), row("c1")])
    assert [r["id"] for r in db.load_chunks("alice", "b1")] == ["c1", "c2"]
    assert db.load_chunks("bob", "b1") == []
    db.delete_chunks("bob", "b1")
    assert len(db.load_chunks("alice", "b1")) == 2
    db.delete_book("alice", "b1")
    assert db.load_chunks("alice", "b1") == []


def test_inserting_chunks_for_a_missing_or_foreign_book_fails(db):
    db.insert_book(book())
    with pytest.raises(BackendError):
        db.insert_chunks([row("c1", book_id="nope")])
    with pytest.raises(BackendError):
        db.insert_chunks([row("c1", user="bob")])
    with pytest.raises(BackendError):  # all-or-nothing
        db.insert_chunks([row("c1"), row("c2", book_id="nope")])
    assert db.load_chunks("alice", "b1") == []


def test_update_book_sets_fields_and_rejects_unknown_ones(db):
    db.insert_book(book())
    before = db.get_book("alice", "b1").updated_at
    db.update_book("alice", "b1", {"status": "ready", "chunks": 7})
    after = db.get_book("alice", "b1")
    assert (after.status, after.chunks) == ("ready", 7) and after.updated_at >= before
    with pytest.raises(ValueError):
        db.update_book("alice", "b1", {"user_id": "bob"})


def test_totals_and_failing_interrupted_books(db):
    db.insert_book(book("b1", size=100))
    db.insert_book(book("b2", user="bob", status="ready", size=50))
    db.insert_chunks([row("c1")])
    assert db.book_totals() == (2, 150)
    assert db.database_bytes() > 0
    assert db.fail_processing_books("interrupted") == 1
    failed = db.get_book("alice", "b1")
    assert (failed.status, failed.error) == ("failed", "interrupted")
    assert db.load_chunks("alice", "b1") == []
    assert db.get_book("bob", "b2").status == "ready"


def test_usage_counts_per_user_and_day(db):
    day = date(2026, 10, 15)
    assert db.usage("alice", day) == 0
    assert db.increment_usage("alice", day) == 1
    assert db.increment_usage("alice", day) == 2
    assert db.usage("bob", day) == 0
    assert db.usage("alice", date(2026, 10, 16)) == 0


def test_user_keys(db):
    assert db.get_user_key("alice") is None
    db.set_user_key("alice", "enc1")
    db.set_user_key("alice", "enc2")
    assert db.get_user_key("alice") == "enc2" and db.get_user_key("bob") is None
    db.delete_user_key("alice")
    assert db.get_user_key("alice") is None


def test_storage_roundtrip():
    s = memory_backend().storage
    s.upload("alice/b1.pdf", b"%PDF")
    assert s.download("alice/b1.pdf") == b"%PDF"
    s.remove("alice/b1.pdf")
    s.remove("alice/b1.pdf")
    with pytest.raises(BackendError):
        s.download("alice/b1.pdf")


def test_auth():
    a = memory_backend().auth
    uid = a.create_user("a@x.org", "password1")
    assert a.verify_password("a@x.org", "password1") == uid
    with pytest.raises(AuthFailed) as e:
        a.create_user("a@x.org", "password2")
    assert e.value.code == "exists"
    with pytest.raises(AuthFailed) as e:
        a.verify_password("a@x.org", "wrong")
    assert e.value.code == "invalid_credentials"
    with pytest.raises(AuthFailed):
        a.verify_password("nobody@x.org", "password1")


def test_book_row_roundtrip():
    b = book()
    assert Book.from_row(b.to_row()) == b
    assert isinstance(b.to_row()["created_at"], str)
