"""In-memory backend: same contracts as Supabase, nothing persisted. Used by tests and by
MEDRAG_BACKEND=memory for trying the multi-user app without a Supabase project."""
import hashlib
import hmac
import os
import threading
import uuid
from dataclasses import replace
from datetime import date

from medrag.cloud.base import AuthFailed, Backend, BackendError, Book, check_update_fields, utc_now


class MemoryDatabase:
    def __init__(self):
        self._lock = threading.RLock()
        self._books: dict[str, Book] = {}
        self._chunks: dict[str, dict] = {}
        self._usage: dict[tuple[str, date], int] = {}
        self._keys: dict[str, str] = {}

    def insert_book(self, book: Book) -> None:
        with self._lock:
            if book.id in self._books:
                raise BackendError(f"book {book.id} already exists")
            self._books[book.id] = book

    def get_book(self, user_id: str, book_id: str) -> Book | None:
        with self._lock:
            book = self._books.get(book_id)
            return book if book is not None and book.user_id == user_id else None

    def list_books(self, user_id: str) -> list[Book]:
        with self._lock:
            return sorted((b for b in self._books.values() if b.user_id == user_id), key=lambda b: b.created_at)

    def update_book(self, user_id: str, book_id: str, changes: dict) -> None:
        check_update_fields(changes)
        with self._lock:
            book = self.get_book(user_id, book_id)
            if book is not None:
                self._books[book_id] = replace(book, **changes, updated_at=utc_now())

    def delete_book(self, user_id: str, book_id: str) -> None:
        with self._lock:
            if self.get_book(user_id, book_id) is not None:
                del self._books[book_id]
                self._drop_chunks(book_id)

    def book_totals(self) -> tuple[int, int]:
        with self._lock:
            return len(self._books), sum(b.size_bytes for b in self._books.values())

    def database_bytes(self) -> int:
        with self._lock:
            return sum(len(r["text"]) + 4 * len(r["embedding"]) for r in self._chunks.values())

    def fail_processing_books(self, reason: str) -> int:
        with self._lock:
            stuck = [b for b in self._books.values() if b.status == "processing"]
            for b in stuck:
                self._drop_chunks(b.id)
                self._books[b.id] = replace(b, status="failed", error=reason, progress="", updated_at=utc_now())
            return len(stuck)

    def insert_chunks(self, rows: list[dict]) -> None:
        with self._lock:
            for r in rows:  # validate everything first: all-or-nothing, like one SQL insert
                if self.get_book(r["user_id"], r["book_id"]) is None:
                    raise BackendError(f"book {r['book_id']} does not exist")
                if r["id"] in self._chunks:
                    raise BackendError(f"chunk {r['id']} already exists")
            for r in rows:
                self._chunks[r["id"]] = dict(r)

    def delete_chunks(self, user_id: str, book_id: str) -> None:
        with self._lock:
            for cid in [k for k, r in self._chunks.items() if r["book_id"] == book_id and r["user_id"] == user_id]:
                del self._chunks[cid]

    def load_chunks(self, user_id: str, book_id: str) -> list[dict]:
        with self._lock:
            rows = [dict(r) for r in self._chunks.values() if r["book_id"] == book_id and r["user_id"] == user_id]
            return sorted(rows, key=lambda r: r["id"])

    def usage(self, user_id: str, day: date) -> int:
        with self._lock:
            return self._usage.get((user_id, day), 0)

    def increment_usage(self, user_id: str, day: date) -> int:
        with self._lock:
            self._usage[(user_id, day)] = self._usage.get((user_id, day), 0) + 1
            return self._usage[(user_id, day)]

    def get_user_key(self, user_id: str) -> str | None:
        with self._lock:
            return self._keys.get(user_id)

    def set_user_key(self, user_id: str, encrypted: str) -> None:
        with self._lock:
            self._keys[user_id] = encrypted

    def delete_user_key(self, user_id: str) -> None:
        with self._lock:
            self._keys.pop(user_id, None)

    def _drop_chunks(self, book_id: str) -> None:
        for cid in [k for k, r in self._chunks.items() if r["book_id"] == book_id]:
            del self._chunks[cid]


class MemoryStorage:
    def __init__(self):
        self._lock = threading.Lock()
        self._files: dict[str, bytes] = {}

    def upload(self, path: str, data: bytes) -> None:
        with self._lock:
            self._files[path] = bytes(data)

    def download(self, path: str) -> bytes:
        with self._lock:
            if path not in self._files:
                raise BackendError(f"object not found: {path}")
            return self._files[path]

    def remove(self, path: str) -> None:
        with self._lock:
            self._files.pop(path, None)


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)


class MemoryAuth:
    def __init__(self):
        self._lock = threading.Lock()
        self._users: dict[str, tuple[str, bytes, bytes]] = {}  # email -> (id, salt, hash)

    def create_user(self, email: str, password: str) -> str:
        with self._lock:
            if email in self._users:
                raise AuthFailed("exists", "User already registered")
            salt = os.urandom(16)
            user_id = str(uuid.uuid4())
            self._users[email] = (user_id, salt, _hash(password, salt))
            return user_id

    def verify_password(self, email: str, password: str) -> str:
        with self._lock:
            record = self._users.get(email)
        if record is None or not hmac.compare_digest(record[2], _hash(password, record[1])):
            raise AuthFailed("invalid_credentials", "Invalid login credentials")
        return record[0]


def memory_backend() -> Backend:
    return Backend(db=MemoryDatabase(), storage=MemoryStorage(), auth=MemoryAuth())
