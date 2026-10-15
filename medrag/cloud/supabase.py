"""Supabase implementation of the cloud contracts. The data client uses the service-role key,
which bypasses RLS, so every query here filters by user_id explicitly. Sign-in uses a fresh
anon-key client per call so the data client's credentials are never replaced."""
import functools
from datetime import date

import httpx

from medrag.cloud.base import (
    AuthFailed, Backend, BackendError, BackendUnavailable, Book, check_update_fields, utc_now,
)

BUCKET = "books"
PAGE_ROWS = 1000  # PostgREST's default max rows per request
INSERT_ROWS = 200
CHUNK_COLUMNS = "id,book_id,user_id,text,page_index,page_label,section,topic,embedding"


def guard(fn):
    """Network trouble or a 5xx (paused / waking project) -> BackendUnavailable; any other
    store failure -> BackendError. Our own errors pass through."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (BackendError, BackendUnavailable, AuthFailed):
            raise
        except (httpx.TransportError, ConnectionError, TimeoutError) as exc:
            raise BackendUnavailable(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise BackendUnavailable(str(exc)) from exc
            raise BackendError(str(exc)) from exc
        except Exception as exc:
            if _is_outage(exc):
                raise BackendUnavailable(str(exc)) from exc
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


_PGRST_CONNECTION_CODES = {"PGRST000", "PGRST001", "PGRST002", "PGRST003"}  # database unreachable / starting


def _is_outage(exc: Exception) -> bool:
    """postgrest raises APIError for every non-2xx response (code = the HTTP status as an int
    when the body isn't JSON, e.g. a gateway 5xx); storage3 raises StorageApiError with .status."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code >= 500
    if code in _PGRST_CONNECTION_CODES:
        return True
    status = getattr(exc, "status", None)
    try:
        return status is not None and int(status) >= 500
    except (TypeError, ValueError):
        return False


def auth_failure(exc: Exception) -> AuthFailed:
    code = str(getattr(exc, "code", "") or "")
    text = str(exc).lower()
    if code in ("user_already_exists", "email_exists") or "already been registered" in text or "already registered" in text:
        return AuthFailed("exists", str(exc))
    if code == "invalid_credentials" or "invalid login" in text:
        return AuthFailed("invalid_credentials", str(exc))
    if code == "weak_password" or ("password" in text and ("at least" in text or "weak" in text)):
        return AuthFailed("weak_password", str(exc))
    return AuthFailed("other", str(exc))


class SupabaseDatabase:
    def __init__(self, client):
        self._c = client

    def _books(self):
        return self._c.table("books")

    def _chunks(self):
        return self._c.table("chunks")

    @guard
    def insert_book(self, book: Book) -> None:
        self._books().insert(book.to_row()).execute()

    @guard
    def get_book(self, user_id: str, book_id: str) -> Book | None:
        rows = self._books().select("*").eq("user_id", user_id).eq("id", book_id).execute().data
        return Book.from_row(rows[0]) if rows else None

    @guard
    def list_books(self, user_id: str) -> list[Book]:
        rows = self._books().select("*").eq("user_id", user_id).order("created_at").execute().data
        return [Book.from_row(r) for r in rows]

    @guard
    def update_book(self, user_id: str, book_id: str, changes: dict, expect_status: str | None = None) -> bool:
        check_update_fields(changes)
        payload = {**changes, "updated_at": utc_now().isoformat()}
        query = self._books().update(payload).eq("user_id", user_id).eq("id", book_id)
        if expect_status is not None:
            query = query.eq("status", expect_status)
        return bool(query.execute().data)  # PostgREST returns the updated rows

    @guard
    def delete_book(self, user_id: str, book_id: str) -> None:
        self._books().delete().eq("user_id", user_id).eq("id", book_id).execute()

    @guard
    def book_totals(self) -> tuple[int, int]:
        rows = self._books().select("size_bytes").execute().data
        return len(rows), sum(int(r["size_bytes"]) for r in rows)

    @guard
    def database_bytes(self) -> int:
        return int(self._c.rpc("database_bytes", {}).execute().data)

    @guard
    def fail_processing_books(self, reason: str) -> int:
        ids = [r["id"] for r in self._books().select("id").eq("status", "processing").execute().data]
        if ids:
            self._chunks().delete().in_("book_id", ids).execute()
            self._books().update(
                {"status": "failed", "error": reason, "progress": "", "updated_at": utc_now().isoformat()}
            ).in_("id", ids).execute()
        return len(ids)

    @guard
    def insert_chunks(self, rows: list[dict]) -> None:
        for i in range(0, len(rows), INSERT_ROWS):
            self._chunks().insert(rows[i : i + INSERT_ROWS]).execute()

    @guard
    def delete_chunks(self, user_id: str, book_id: str) -> None:
        self._chunks().delete().eq("user_id", user_id).eq("book_id", book_id).execute()

    @guard
    def load_chunks(self, user_id: str, book_id: str) -> list[dict]:
        out: list[dict] = []
        while True:
            part = (
                self._chunks().select(CHUNK_COLUMNS).eq("user_id", user_id).eq("book_id", book_id)
                .order("id").range(len(out), len(out) + PAGE_ROWS - 1).execute().data
            )
            out.extend(part)
            if len(part) < PAGE_ROWS:
                return out

    @guard
    def usage(self, user_id: str, day: date) -> int:
        rows = (
            self._c.table("usage").select("shared_key_questions")
            .eq("user_id", user_id).eq("day", day.isoformat()).execute().data
        )
        return int(rows[0]["shared_key_questions"]) if rows else 0

    @guard
    def increment_usage(self, user_id: str, day: date) -> int:
        return int(self._c.rpc("increment_usage", {"p_user": user_id, "p_day": day.isoformat()}).execute().data)

    @guard
    def get_user_key(self, user_id: str) -> str | None:
        rows = self._c.table("user_keys").select("encrypted_groq_key").eq("user_id", user_id).execute().data
        return rows[0]["encrypted_groq_key"] if rows else None

    @guard
    def set_user_key(self, user_id: str, encrypted: str) -> None:
        self._c.table("user_keys").upsert({"user_id": user_id, "encrypted_groq_key": encrypted}).execute()

    @guard
    def delete_user_key(self, user_id: str) -> None:
        self._c.table("user_keys").delete().eq("user_id", user_id).execute()


class SupabaseStorage:
    def __init__(self, client):
        self._c = client

    @guard
    def upload(self, path: str, data: bytes) -> None:
        self._c.storage.from_(BUCKET).upload(path, data, {"content-type": "application/pdf"})

    @guard
    def download(self, path: str) -> bytes:
        return self._c.storage.from_(BUCKET).download(path)

    @guard
    def remove(self, path: str) -> None:
        self._c.storage.from_(BUCKET).remove([path])


def _client(url: str, key: str):
    from supabase import ClientOptions, create_client

    return create_client(url, key, options=ClientOptions(auto_refresh_token=False, persist_session=False))


class SupabaseAuth:
    def __init__(self, admin_client, url: str, anon_key: str):
        self._admin = admin_client
        self._url = url
        self._anon_key = anon_key

    def create_user(self, email: str, password: str) -> str:
        try:
            res = self._admin.auth.admin.create_user({"email": email, "password": password, "email_confirm": True})
        except (httpx.TransportError, ConnectionError, TimeoutError) as exc:
            raise BackendUnavailable(str(exc)) from exc
        except Exception as exc:
            raise auth_failure(exc) from exc
        return str(res.user.id)

    def verify_password(self, email: str, password: str) -> str:
        client = _client(self._url, self._anon_key)
        try:
            res = client.auth.sign_in_with_password({"email": email, "password": password})
        except (httpx.TransportError, ConnectionError, TimeoutError) as exc:
            raise BackendUnavailable(str(exc)) from exc
        except Exception as exc:
            raise auth_failure(exc) from exc
        try:
            client.auth.sign_out()
        except Exception:
            pass
        return str(res.user.id)

    def delete_user(self, user_id: str) -> None:
        """Test cleanup only."""
        self._admin.auth.admin.delete_user(user_id)


def supabase_backend(url: str, service_key: str, anon_key: str) -> Backend:
    admin = _client(url, service_key)
    return Backend(db=SupabaseDatabase(admin), storage=SupabaseStorage(admin), auth=SupabaseAuth(admin, url, anon_key))
