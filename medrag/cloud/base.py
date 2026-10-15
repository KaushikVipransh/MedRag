"""Contracts between MedRAG and its durable store, plus the Book record. Two implementations:
memory.py (tests and development) and supabase.py (production)."""
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from typing import Protocol


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BackendError(RuntimeError):
    """The store refused an operation (missing row, constraint violation, ...)."""


class BackendUnavailable(RuntimeError):
    """The store could not be reached (network error, project paused or waking up)."""


class AuthFailed(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code  # "exists" | "invalid_credentials" | "weak_password" | "other"


@dataclass(frozen=True)
class Book:
    id: str
    user_id: str
    title: str
    filename: str
    storage_path: str
    size_bytes: int
    pages: int = 0
    chunks: int = 0
    status: str = "processing"  # processing | ready | failed
    progress: str = ""
    error: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def to_row(self) -> dict:
        row = asdict(self)
        row["created_at"] = self.created_at.isoformat()
        row["updated_at"] = self.updated_at.isoformat()
        return row

    @classmethod
    def from_row(cls, row: dict) -> "Book":
        values = {f.name: row[f.name] for f in fields(cls) if f.name in row}
        for name in ("created_at", "updated_at"):
            if isinstance(values.get(name), str):
                values[name] = datetime.fromisoformat(values[name])
        for name in ("size_bytes", "pages", "chunks"):
            if name in values:
                values[name] = int(values[name] or 0)
        for name in ("progress", "error"):
            values[name] = values.get(name) or ""
        return cls(**values)


UPDATABLE_BOOK_FIELDS = frozenset({"status", "progress", "error", "pages", "chunks"})


def check_update_fields(changes: dict) -> None:
    unknown = set(changes) - UPDATABLE_BOOK_FIELDS
    if unknown:
        raise ValueError(f"book fields cannot be updated: {sorted(unknown)}")


class Database(Protocol):
    def insert_book(self, book: Book) -> None: ...
    def get_book(self, user_id: str, book_id: str) -> Book | None: ...
    def list_books(self, user_id: str) -> list[Book]: ...
    def update_book(self, user_id: str, book_id: str, changes: dict, expect_status: str | None = None) -> bool: ...
    def delete_book(self, user_id: str, book_id: str) -> None: ...
    def book_totals(self) -> tuple[int, int]: ...
    def database_bytes(self) -> int: ...
    def fail_processing_books(self, reason: str) -> int: ...
    def insert_chunks(self, rows: list[dict]) -> None: ...
    def delete_chunks(self, user_id: str, book_id: str) -> None: ...
    def load_chunks(self, user_id: str, book_id: str) -> list[dict]: ...
    def usage(self, user_id: str, day: date) -> int: ...
    def increment_usage(self, user_id: str, day: date) -> int: ...
    def get_user_key(self, user_id: str) -> str | None: ...
    def set_user_key(self, user_id: str, encrypted: str) -> None: ...
    def delete_user_key(self, user_id: str) -> None: ...


class Storage(Protocol):
    def upload(self, path: str, data: bytes) -> None: ...
    def download(self, path: str) -> bytes: ...
    def remove(self, path: str) -> None: ...


class Auth(Protocol):
    def create_user(self, email: str, password: str) -> str: ...
    def verify_password(self, email: str, password: str) -> str: ...


@dataclass
class Backend:
    db: Database
    storage: Storage
    auth: Auth
