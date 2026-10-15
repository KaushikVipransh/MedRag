"""A user's private books: upload limits, storage, status and deletion. Every method takes
user_id; the store filters on it, so one user can never see or touch another's books."""
from __future__ import annotations

import uuid
from pathlib import PurePath
from typing import Callable

from medrag.cloud.base import Backend, Book
from medrag.config import CloudSettings
from medrag.ingest.clean import strip_control
from medrag.ingest.precheck import PdfCheck, PdfRejected, check_pdf
from medrag.store_pg import chunk_row
from medrag.types import Chunk

MB = 1024 * 1024
MAX_TITLE = 200
INTERRUPTED_MESSAGE = "Indexing was interrupted because the app restarted. Delete the book and upload it again."


class UploadRejected(ValueError):
    """str(error) is safe to show to the user."""


def clean_title(title: str, filename: str) -> str:
    title, filename = strip_control(title), strip_control(filename)
    cleaned = " ".join(title.split())
    if not cleaned:
        stem = PurePath(filename).stem if not filename.startswith(".") else ""
        cleaned = " ".join(stem.replace("_", " ").replace("-", " ").split())
    return cleaned[:MAX_TITLE] or "Untitled book"


class Library:
    def __init__(self, backend: Backend, cloud: CloudSettings, new_id: Callable[[], str] = lambda: str(uuid.uuid4())):
        self.db = backend.db
        self.storage = backend.storage
        self.cloud = cloud
        self.new_id = new_id
        self.on_delete: list[Callable[[str], None]] = []

    def list(self, user_id: str) -> list[Book]:
        return self.db.list_books(user_id)

    def get(self, user_id: str, book_id: str) -> Book | None:
        return self.db.get_book(user_id, book_id)

    def check_upload(self, user_id: str, data: bytes) -> PdfCheck:
        c = self.cloud
        mine = len(self.db.list_books(user_id))
        if mine >= c.max_books_per_user:
            raise UploadRejected(
                f"You can keep up to {c.max_books_per_user} books (you have {mine}). Delete one to upload another."
            )
        count, total_bytes = self.db.book_totals()
        if count >= c.global_book_cap:
            raise UploadRejected(
                f"MedRAG has reached its limit of {c.global_book_cap} books across all users, "
                "so uploads are paused. Please try again later."
            )
        if total_bytes + len(data) > c.storage_cap_mb * MB:
            raise UploadRejected(
                f"MedRAG's free file storage is full ({total_bytes / MB:.0f} of {c.storage_cap_mb} MB used), "
                "so uploads are paused. Please try again later."
            )
        if self.db.database_bytes() > c.db_cap_mb * MB:
            raise UploadRejected("MedRAG's free database is nearly full, so uploads are paused. Please try again later.")
        try:
            return check_pdf(data, max_bytes=c.max_pdf_mb * MB, max_pages=c.max_pages)
        except PdfRejected as exc:
            raise UploadRejected(str(exc)) from exc

    def add(self, user_id: str, title: str, filename: str, data: bytes) -> Book:
        check = self.check_upload(user_id, data)
        book_id = self.new_id()
        book = Book(
            id=book_id, user_id=user_id, title=clean_title(title, filename), filename=strip_control(filename)[:255],
            storage_path=f"{user_id}/{book_id}.pdf", size_bytes=len(data), pages=check.pages,
            status="processing", progress="Queued",
        )
        self.storage.upload(book.storage_path, data)
        try:
            self.db.insert_book(book)
        except Exception:
            self.storage.remove(book.storage_path)
            raise
        return book

    def set_progress(self, user_id: str, book_id: str, text: str) -> None:
        self.db.update_book(user_id, book_id, {"progress": text})

    def insert_chunks(self, user_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.db.insert_chunks([chunk_row(c, e, user_id) for c, e in zip(chunks, embeddings)])

    def mark_ready(self, user_id: str, book_id: str, pages: int, chunks: int) -> bool:
        """Only a book still 'processing' becomes ready: if another process failed it meanwhile
        (startup sweep) or it was deleted, this returns False and the caller discards its chunks."""
        return self.db.update_book(
            user_id, book_id, {"status": "ready", "pages": pages, "chunks": chunks, "progress": "", "error": ""},
            expect_status="processing",
        )

    def discard_chunks(self, user_id: str, book_id: str) -> None:
        self.db.delete_chunks(user_id, book_id)

    def mark_failed(self, user_id: str, book_id: str, reason: str) -> None:
        self.db.delete_chunks(user_id, book_id)
        self.db.update_book(user_id, book_id, {"status": "failed", "error": reason, "progress": ""})

    def delete(self, user_id: str, book_id: str) -> bool:
        book = self.get(user_id, book_id)
        if book is None:
            return False
        self.db.delete_book(user_id, book_id)
        self.storage.remove(book.storage_path)
        for callback in self.on_delete:
            callback(book_id)
        return True

    def fail_interrupted(self) -> int:
        """Run once at startup: the ingest queue lives in this process, so any book still
        'processing' belongs to a previous process that stopped mid-way."""
        return self.db.fail_processing_books(INTERRUPTED_MESSAGE)
