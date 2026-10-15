# MedRAG v3 Multi-user Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the single-book MedRAG into a free-tier multi-user app: people sign up, upload their own medical textbook (text PDF), and chat with a bot that answers only from that book, with page citations.

**Architecture:** Streamlit stays the only server. Durable data (accounts, PDFs, chunk rows with embeddings, usage, encrypted Groq keys) lives in Supabase behind three small contracts (`Database`, `Storage`, `Auth`) with two implementations: an in-memory one (tests and `memory` dev mode) and a Supabase one (production). At question time one book's chunks are loaded into an in-memory vector store and searched with today's unchanged retrieval stack (dense + BM25 → RRF → bge re-rank, MiniLM gate). A single background thread ingests uploads one at a time.

**Tech Stack:** Python 3.12, Streamlit 1.65, PyMuPDF, sentence-transformers / transformers (CPU torch), rank-bm25, numpy, Groq SDK, `supabase` (supabase-py v2), `cryptography` (Fernet), pytest + Streamlit AppTest, Docker (Hugging Face Docker Space).

**Spec:** `docs/superpowers/specs/2026-10-15-medrag-multiuser-design.md`

## Global Constraints

- All commands run in the dev container: `docker compose run --rm dev <command>`. In Git Bash prefix container paths with `MSYS_NO_PATHCONV=1`. The host (Windows ARM64, Smart App Control) cannot run the Python stack natively.
- Free tiers only: Hugging Face free CPU Space, Supabase free tier, Groq free tier. Nothing that costs money.
- The GitHub repo is PUBLIC. Never commit PDFs, `.env`, the Chroma index, `data/eval/evidence.jsonl`, `data/eval_cache/`, `data/eval_results/`. Never print API keys in logs, tests or output.
- Limits (exact values from the spec): 3 books per user; PDF ≤ 50 MB and ≤ 1,000 pages; global cap 60 books; total stored PDF bytes ≤ 900 MB; 10 shared-key answers per user per day; in-memory cache of 4 books; a PDF is "scanned" when ≥ 20% of its pages yield no usable text.
- User-facing messages are copied verbatim from the spec where the spec gives them:
  - `This PDF has no text layer (it looks scanned). Upload a text-based PDF.`
  - `This file couldn't be opened as a PDF.`
  - `You've used today's 10 free questions. Add your free Groq key in Settings to keep going.` (the number comes from the cap setting)
  - `The free shared quota is used up for today. Add your free Groq key in Settings.`
- Every data-access method that touches user data takes `user_id` and filters by it. The service-role key bypasses RLS, so this filter is the real isolation.
- Models: generator `openai/gpt-oss-120b`; embedder `sentence-transformers/all-MiniLM-L6-v2` (384-dim); re-ranker `BAAI/bge-reranker-base`; gate `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Local mode keeps working: with no Supabase settings and no `MEDRAG_BACKEND`, `streamlit run app.py` is today's single-book app over Chroma, and the evaluation harness is untouched.
- `pytest` (default, excludes `slow`) must stay green after every task. Current baseline: 201 passed.

## Review Focus

1. **A book deleted while it is still being indexed:** no orphan chunk rows remain and the worker keeps processing the queue. Test in Task 10.
2. **Asking about a book that isn't yours, isn't ready, or was just deleted** (stale selection in another tab): a polite "not available" answer with no LLM call and no quota use. Test in Task 11.
3. **One ingestion job blowing up** (missing storage object, backend error) must not kill the single worker thread; later uploads still get indexed. Test in Task 10.
4. **UTC midnight rollover and per-user counting:** the shared cap resets on a new UTC day, and one user's usage never affects another's. Test in Task 6.
5. **Blank, whitespace-only or very long titles, and odd filenames** (`My Book (2).PDF`, unicode) get a sensible title; uploading the same file twice creates two separate books. Test in Task 9.

## Deviations from the spec (decided while planning; record them in the spec in Task 15)

- **Anon key added.** Sign-in uses a short-lived client built with `SUPABASE_ANON_KEY`, because signing in on the service-role client would replace its auth header. Sign-up uses the admin API (`auth.admin.create_user` with `email_confirm: True`), so no confirmation email is ever needed.
- **Interrupted jobs are swept at startup, not after 15 minutes.** The queue lives in one process, so at process start every `processing` row is an orphan and is marked failed ("interrupted") immediately. A 15-minute timer would wrongly fail books that are merely queued behind a long ingestion.
- **Database-size guard.** Besides the 60-book cap, uploads pause when Postgres reports more than 400 MB used (`database_bytes()` RPC), because chunk size varies between books.
- **`Dockerfile`, not `Dockerfile.prod`.** Hugging Face Docker Spaces only build a file named `Dockerfile` at the repo root.
- **Free-tier facts corrected:** a paused Supabase project does not wake on request; it is restored from the dashboard (free). A free Space sleeps after 48 hours without visitors.
- **Two extra `books` columns:** `size_bytes` (for the 900 MB storage cap) and `updated_at`.
- **`InMemoryChunkStore` is read-only** (`count`, `all_chunks`, `documents`, `query`); writes go through the `Database` contract. The retriever only needs `query`.
- **Tests fake the contracts, not the Supabase client.** The in-memory backend is the fake; Supabase-specific code gets unit tests for its error mapping plus the live test.

## File structure

| File | Responsibility |
|---|---|
| `medrag/config.py` (modify) | `CloudSettings`: backend kind, Supabase settings, limits |
| `medrag/generation/llm.py` (modify) | `LLMRateLimited`, `LLMAuthError` subclasses of `LLMError` |
| `medrag/generation/generator.py` (modify) | Map LLM failures to `rate_limited` / `key_rejected` / `llm_error` |
| `medrag/cloud/__init__.py` | Package marker |
| `medrag/cloud/base.py` | `Book`, errors, `Database` / `Storage` / `Auth` protocols, `Backend` |
| `medrag/cloud/memory.py` | In-memory implementations (tests, `memory` mode) |
| `medrag/cloud/supabase.py` | Supabase implementations |
| `medrag/accounts.py` | Sign up / sign in with validation and friendly errors |
| `medrag/keys.py` | `KeyVault`: Fernet-encrypted user Groq keys, masking, validation |
| `medrag/quota.py` | Shared-key daily cap and key choice |
| `medrag/ingest/pdf.py` (modify) | Read PDFs from a path or bytes |
| `medrag/ingest/build.py` (modify) | `chunks_for_pdf()` shared by CLI ingestion and the worker |
| `medrag/ingest/precheck.py` | Upload pre-checks: opens, size, pages, text layer |
| `medrag/store_pg.py` | Chunk ↔ row conversion and `InMemoryChunkStore` |
| `medrag/library.py` | A user's books: limits, add, list, delete, status changes |
| `medrag/ingest/job.py` | `IngestWorker`: background, one at a time |
| `medrag/chat.py` | `ChatService.ask()`: ownership, key choice, fallback, counting, retriever LRU |
| `medrag/services.py` | Wiring: `Models`, `Services`, `assemble`, `build_services`, `upload_book`, `delete_book` |
| `medrag/web/__init__.py`, `components.py`, `local.py`, `multiuser.py` | Streamlit UI (local mode moved out of `app.py`; new multi-user pages) |
| `app.py` (modify) | Chooses local or multi-user UI |
| `supabase/schema.sql` | Tables, indexes, RLS, RPCs, bucket |
| `Dockerfile`, `scripts/download_models.py`, `.streamlit/config.toml` | Production image |
| `docs/deploy.md`, `README.md` | Deployment guide |
| `tests/fakes_cloud.py` | Test PDFs, keyed fake LLM factory, `make_services()` |

---

### Task 0: Prerequisites carried over from v2-accuracy

**Files:**
- Modify: `medrag/config.py:20`
- Modify: `tests/test_config.py:33`
- Create: `docs/notes/dev-subset-check.md`

**Interfaces:**
- Produces: `Settings().rerank_threshold == 0.045`

- [ ] **Step 1: Update the test to the calibrated value**

In `tests/test_config.py`, `test_retrieval_model_defaults`, change the last assertion to:

```python
    assert s.rerank_threshold == 0.045
```

- [ ] **Step 2: Run it to see it fail**

Run: `docker compose run --rm dev pytest tests/test_config.py -q`
Expected: FAIL, `assert 0.06 == 0.045`

- [ ] **Step 3: Lower the threshold**

In `medrag/config.py` replace the `rerank_threshold` line with:

```python
    rerank_threshold: float = 0.045  # minimum gate relevance (sigmoid): weakest verified in-scope 0.057, strongest out-of-scope 0.037
```

- [ ] **Step 4: Run the full suite**

Run: `docker compose run --rm dev pytest -q`
Expected: 201 passed

- [ ] **Step 5: Run the 40-item dev-subset check (needs Groq quota)**

Run: `docker compose run --rm dev python -m medrag.eval run --systems medrag --ids-file data/eval/dev_subset.txt`
then: `docker compose run --rm dev python -m medrag.eval report`

Expected: the run finishes; the report shows in-scope correctness at least the baseline (90.5% of graded) and 100% handling for out-of-scope / near-miss / adversarial. If it aborts with `QuotaExhausted`, the run is resumable: write that in the note and continue with Task 1. Do not block the plan on quota.

Write `docs/notes/dev-subset-check.md` with: the date, the threshold (0.045), the commands, and the report's summary table (or "pending: Groq daily quota exhausted on <date>"). Do not paste any API key or book text beyond the report's own figures.

- [ ] **Step 6: Commit**

```bash
git add medrag/config.py tests/test_config.py docs/notes/dev-subset-check.md
git commit -m "fix(retrieval): lower relevance threshold to 0.045 per calibration; record dev-subset check"
```

---

### Task 1: Cloud settings and dependencies

**Files:**
- Modify: `medrag/config.py`
- Modify: `requirements.in`, `requirements.txt`, `.env.example`
- Create: `.streamlit/config.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `CloudSettings` (frozen dataclass) with fields `backend: str` (`"local" | "memory" | "supabase"`), `supabase_url`, `supabase_anon_key`, `supabase_service_key`, `fernet_key` (all `str`), `shared_daily_cap=10`, `max_books_per_user=3`, `max_pdf_mb=50`, `max_pages=1000`, `global_book_cap=60`, `storage_cap_mb=900`, `db_cap_mb=400`, `cached_books=4`; `CloudSettings.from_env(load_dotenv_file=True)`; `CloudSettings.missing() -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
from medrag.config import CloudSettings

CLOUD_VARS = ("MEDRAG_BACKEND", "SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY", "MEDRAG_FERNET_KEY")


def _clear_cloud(monkeypatch):
    for name in CLOUD_VARS:
        monkeypatch.delenv(name, raising=False)


def test_cloud_defaults_to_local_without_supabase(monkeypatch):
    _clear_cloud(monkeypatch)
    c = CloudSettings.from_env(load_dotenv_file=False)
    assert c.backend == "local"
    assert (c.shared_daily_cap, c.max_books_per_user, c.max_pdf_mb, c.max_pages) == (10, 3, 50, 1000)
    assert (c.global_book_cap, c.storage_cap_mb, c.db_cap_mb, c.cached_books) == (60, 900, 400, 4)


def test_cloud_picks_supabase_when_url_is_set(monkeypatch):
    _clear_cloud(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    c = CloudSettings.from_env(load_dotenv_file=False)
    assert c.backend == "supabase"
    assert c.missing() == ["SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY", "MEDRAG_FERNET_KEY"]


def test_explicit_backend_wins(monkeypatch):
    _clear_cloud(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("MEDRAG_BACKEND", "memory")
    c = CloudSettings.from_env(load_dotenv_file=False)
    assert c.backend == "memory" and c.missing() == []


def test_cloud_limits_from_env(monkeypatch):
    _clear_cloud(monkeypatch)
    monkeypatch.setenv("MEDRAG_SHARED_DAILY_CAP", "25")
    monkeypatch.setenv("MEDRAG_GLOBAL_BOOK_CAP", "5")
    c = CloudSettings.from_env(load_dotenv_file=False)
    assert c.shared_daily_cap == 25 and c.global_book_cap == 5
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_config.py -q`
Expected: FAIL with `ImportError: cannot import name 'CloudSettings'`

- [ ] **Step 3: Implement `CloudSettings`**

Append to `medrag/config.py`:

```python
@dataclass(frozen=True)
class CloudSettings:
    """Multi-user settings. backend: "local" (single-book Chroma app), "memory" (multi-user,
    nothing persisted; for development and tests) or "supabase" (production)."""

    backend: str = "local"
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""
    fernet_key: str = ""
    shared_daily_cap: int = 10
    max_books_per_user: int = 3
    max_pdf_mb: int = 50
    max_pages: int = 1000
    global_book_cap: int = 60
    storage_cap_mb: int = 900
    db_cap_mb: int = 400
    cached_books: int = 4

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "CloudSettings":
        if load_dotenv_file:
            from dotenv import load_dotenv

            load_dotenv()
        d = cls()
        env = os.getenv
        url = env("SUPABASE_URL", "")
        return cls(
            backend=env("MEDRAG_BACKEND", "") or ("supabase" if url else "local"),
            supabase_url=url,
            supabase_anon_key=env("SUPABASE_ANON_KEY", ""),
            supabase_service_key=env("SUPABASE_SERVICE_KEY", ""),
            fernet_key=env("MEDRAG_FERNET_KEY", ""),
            shared_daily_cap=int(env("MEDRAG_SHARED_DAILY_CAP", d.shared_daily_cap)),
            max_books_per_user=int(env("MEDRAG_MAX_BOOKS_PER_USER", d.max_books_per_user)),
            max_pdf_mb=int(env("MEDRAG_MAX_PDF_MB", d.max_pdf_mb)),
            max_pages=int(env("MEDRAG_MAX_PAGES", d.max_pages)),
            global_book_cap=int(env("MEDRAG_GLOBAL_BOOK_CAP", d.global_book_cap)),
            storage_cap_mb=int(env("MEDRAG_STORAGE_CAP_MB", d.storage_cap_mb)),
            db_cap_mb=int(env("MEDRAG_DB_CAP_MB", d.db_cap_mb)),
            cached_books=int(env("MEDRAG_CACHED_BOOKS", d.cached_books)),
        )

    def missing(self) -> list[str]:
        """Names of required settings that are empty for the chosen backend."""
        if self.backend != "supabase":
            return []
        required = {
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_ANON_KEY": self.supabase_anon_key,
            "SUPABASE_SERVICE_KEY": self.supabase_service_key,
            "MEDRAG_FERNET_KEY": self.fernet_key,
        }
        return [name for name, value in required.items() if not value]
```

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm dev pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Add the dependencies and pin them**

Append two lines to `requirements.in`:

```
supabase
cryptography
```

Install into the dev container and capture the pinned versions (Git Bash):

```bash
docker compose run --rm dev sh -c "pip install -q supabase cryptography && pip freeze" > "$TEMP/freeze.txt"
```

Rebuild `requirements.txt` as: line 1 `--extra-index-url https://download.pytorch.org/whl/cpu`, then the freeze output sorted, keeping `torch==...+cpu` as frozen. Check with `grep -E "^(supabase|cryptography|torch)==" requirements.txt`: all three present, and no `nvidia-` packages (`grep -c nvidia requirements.txt` → 0). Then rebuild the image:

```bash
docker compose build dev
```

Expected: the build succeeds; `docker compose run --rm dev python -c "import supabase, cryptography; print('ok')"` prints `ok`.

- [ ] **Step 6: Streamlit upload limit and `.env.example`**

Create `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 50

[browser]
gatherUsageStats = false
```

Append to `.env.example`:

```
# Multi-user mode (leave unset for the local single-book app)
# MEDRAG_BACKEND=memory            # multi-user without Supabase; nothing is saved
# SUPABASE_URL=https://<project>.supabase.co
# SUPABASE_ANON_KEY=...
# SUPABASE_SERVICE_KEY=...         # server-side only; never expose
# MEDRAG_FERNET_KEY=...            # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# MEDRAG_SHARED_DAILY_CAP=10
```

- [ ] **Step 7: Full suite and commit**

Run: `docker compose run --rm dev pytest -q` → all pass.

```bash
git add medrag/config.py tests/test_config.py requirements.in requirements.txt .env.example .streamlit/config.toml
git commit -m "feat(config): CloudSettings for multi-user mode; add supabase and cryptography"
```

---

### Task 2: Distinguish rate limits and rejected keys

**Files:**
- Modify: `medrag/generation/llm.py`
- Modify: `medrag/generation/generator.py`
- Modify: `medrag/eval/runner.py:67`
- Test: `tests/test_llm.py`, `tests/test_generator.py`

**Interfaces:**
- Produces: `LLMRateLimited(LLMError)`, `LLMAuthError(LLMError)` in `medrag.generation.llm`; `GroqClient.complete` raises them for Groq `RateLimitError` and `AuthenticationError` / `PermissionDeniedError`. `Answer.refusal_reason` is `"rate_limited"` / `"key_rejected"` for those, else `"llm_error"`. `LLM_FAILURE_REASONS = ("llm_error", "rate_limited", "key_rejected")` in `medrag.generation.generator`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
import pytest

from medrag.generation.llm import LLMAuthError, LLMRateLimited


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


def _client_raising(exc):
    client = GroqClient(api_key="gsk_test", model="m")

    class Completions:
        def create(self, **kwargs):
            raise exc

    client._client = type("C", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    return client


def test_rate_limit_is_typed():
    with pytest.raises(LLMRateLimited, match="RateLimitError"):
        _client_raising(RateLimitError("daily tokens")).complete([{"role": "user", "content": "hi"}])


def test_bad_key_is_typed():
    with pytest.raises(LLMAuthError):
        _client_raising(AuthenticationError("invalid api key")).complete([{"role": "user", "content": "hi"}])


def test_other_errors_stay_generic():
    with pytest.raises(LLMError) as info:
        _client_raising(ValueError("boom")).complete([{"role": "user", "content": "hi"}])
    assert type(info.value) is LLMError
```

Append to `tests/test_generator.py`:

```python
from medrag.generation.llm import LLMAuthError, LLMRateLimited


def test_rate_limit_becomes_rate_limited_refusal():
    ans = Generator(FakeLLM(error=LLMRateLimited("RateLimitError: tokens per day"))).generate("q", chunks())
    assert ans.refused and ans.refusal_reason == "rate_limited"


def test_rejected_key_becomes_key_rejected_refusal():
    ans = Generator(FakeLLM(error=LLMAuthError("AuthenticationError: invalid"))).generate("q", chunks())
    assert ans.refused and ans.refusal_reason == "key_rejected"
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_llm.py tests/test_generator.py -q`
Expected: FAIL with `ImportError: cannot import name 'LLMAuthError'`

- [ ] **Step 3: Implement**

In `medrag/generation/llm.py`, add after `class LLMError`:

```python
class LLMRateLimited(LLMError):
    """Groq refused the request because a per-minute or daily limit was hit."""


class LLMAuthError(LLMError):
    """Groq rejected the API key."""
```

and replace the `except` block in `GroqClient.complete` with:

```python
        except Exception as exc:  # network, auth, rate limit, model retired
            message = f"{type(exc).__name__}: {exc}"
            name = type(exc).__name__
            if name == "RateLimitError":
                raise LLMRateLimited(message) from exc
            if name in ("AuthenticationError", "PermissionDeniedError"):
                raise LLMAuthError(message) from exc
            raise LLMError(message) from exc
```

In `medrag/generation/generator.py`, change the import to `from medrag.generation.llm import LLMAuthError, LLMClient, LLMError, LLMRateLimited`, add below `LLM_UNAVAILABLE_MESSAGE`:

```python
LLM_FAILURE_REASONS = ("llm_error", "rate_limited", "key_rejected")


def _failure_reason(exc: LLMError) -> str:
    if isinstance(exc, LLMRateLimited):
        return "rate_limited"
    if isinstance(exc, LLMAuthError):
        return "key_rejected"
    return "llm_error"
```

and in `generate`, change the `except LLMError as exc:` return to use `refusal_reason=_failure_reason(exc)`.

In `medrag/eval/runner.py` line 67, change `if ans.refusal_reason == "llm_error":` to `if ans.refusal_reason in LLM_FAILURE_REASONS:` and import it: `from medrag.generation.generator import LLM_FAILURE_REASONS`.

- [ ] **Step 4: Run the full suite**

Run: `docker compose run --rm dev pytest -q`
Expected: all pass (the existing `test_llm_failure_becomes_friendly_answer` still sees `llm_error` for a plain `LLMError`).

- [ ] **Step 5: Commit**

```bash
git add medrag/generation/llm.py medrag/generation/generator.py medrag/eval/runner.py tests/test_llm.py tests/test_generator.py
git commit -m "feat(llm): typed rate-limit and rejected-key errors, surfaced as refusal reasons"
```

---

### Task 3: Cloud contracts and the in-memory backend

**Files:**
- Create: `medrag/cloud/__init__.py` (empty), `medrag/cloud/base.py`, `medrag/cloud/memory.py`
- Test: `tests/test_cloud_memory.py`

**Interfaces:**
- Produces (in `medrag.cloud.base`):
  - `Book` (frozen dataclass): `id, user_id, title, filename, storage_path: str; size_bytes: int; pages: int = 0; chunks: int = 0; status: str = "processing"; progress: str = ""; error: str = ""; created_at, updated_at: datetime` (UTC); `Book.to_row() -> dict`; `Book.from_row(dict) -> Book`.
  - `UPDATABLE_BOOK_FIELDS = {"status", "progress", "error", "pages", "chunks"}`; `check_update_fields(fields: dict) -> None` raises `ValueError` on others.
  - Errors: `BackendError(RuntimeError)`, `BackendUnavailable(RuntimeError)`, `AuthFailed(Exception)` with `.code` in `{"exists", "invalid_credentials", "weak_password", "other"}`.
  - Protocols `Database`, `Storage`, `Auth` (methods below) and `Backend(db, storage, auth)` dataclass.
  - `Database`: `insert_book(book)`, `get_book(user_id, book_id) -> Book | None`, `list_books(user_id) -> list[Book]` (oldest first), `update_book(user_id, book_id, fields: dict)` (also sets `updated_at`; no-op if missing), `delete_book(user_id, book_id)` (cascades chunks), `book_totals() -> tuple[int, int]` (all users: count, total `size_bytes`), `database_bytes() -> int`, `fail_processing_books(reason: str) -> int`, `insert_chunks(rows: list[dict])` (raises `BackendError` if a row's book does not exist for that user), `delete_chunks(user_id, book_id)`, `load_chunks(user_id, book_id) -> list[dict]` (ordered by id), `usage(user_id, day: date) -> int`, `increment_usage(user_id, day: date) -> int`, `get_user_key(user_id) -> str | None`, `set_user_key(user_id, encrypted: str)`, `delete_user_key(user_id)`.
  - `Storage`: `upload(path: str, data: bytes)`, `download(path: str) -> bytes` (raises `BackendError` if missing), `remove(path: str)` (no error if missing).
  - `Auth`: `create_user(email, password) -> str`, `verify_password(email, password) -> str`; both raise `AuthFailed`.
- Produces (in `medrag.cloud.memory`): `MemoryDatabase`, `MemoryStorage`, `MemoryAuth`, `memory_backend() -> Backend`.
- Chunk row dict keys (used by every later task): `id, book_id, user_id, text, page_index, page_label, section, topic, embedding` (`list[float]`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cloud_memory.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_cloud_memory.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.cloud'`

- [ ] **Step 3: Write `medrag/cloud/base.py`**

```python
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
    def update_book(self, user_id: str, book_id: str, changes: dict) -> None: ...
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
```

- [ ] **Step 4: Write `medrag/cloud/memory.py`**

```python
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
```

- [ ] **Step 5: Run the tests**

Run: `docker compose run --rm dev pytest tests/test_cloud_memory.py -q`
Expected: PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add medrag/cloud tests/test_cloud_memory.py
git commit -m "feat(cloud): storage contracts, Book record and in-memory backend with per-user isolation"
```

---

### Task 4: Accounts

**Files:**
- Create: `medrag/accounts.py`
- Test: `tests/test_accounts.py`

**Interfaces:**
- Consumes: `Auth`, `AuthFailed` from `medrag.cloud.base`; `memory_backend()` in tests.
- Produces: `User(id: str, email: str)` frozen dataclass; `AuthError(ValueError)` whose `str()` is the user-facing message; `Accounts(auth).sign_up(email, password) -> User`, `Accounts(auth).sign_in(email, password) -> User`; `MIN_PASSWORD = 8`. `BackendUnavailable` propagates unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_accounts.py`:

```python
import pytest

from medrag.accounts import MIN_PASSWORD, Accounts, AuthError
from medrag.cloud.base import AuthFailed, BackendUnavailable
from medrag.cloud.memory import MemoryAuth


@pytest.fixture
def accounts():
    return Accounts(MemoryAuth())


def test_sign_up_then_sign_in(accounts):
    user = accounts.sign_up("  Alice@Example.org ", "password1")
    assert user.email == "alice@example.org"
    assert accounts.sign_in("alice@example.org", "password1") == user


def test_duplicate_email(accounts):
    accounts.sign_up("a@x.org", "password1")
    with pytest.raises(AuthError, match="already exists"):
        accounts.sign_up("A@X.org", "password2")


def test_wrong_password(accounts):
    accounts.sign_up("a@x.org", "password1")
    with pytest.raises(AuthError, match="Wrong email or password"):
        accounts.sign_in("a@x.org", "nope-nope")


@pytest.mark.parametrize("email", ["", "no-at-sign", "a@b", "a b@x.org"])
def test_invalid_email(accounts, email):
    with pytest.raises(AuthError, match="valid email"):
        accounts.sign_up(email, "password1")


def test_short_password(accounts):
    with pytest.raises(AuthError, match=str(MIN_PASSWORD)):
        accounts.sign_up("a@x.org", "short")


def test_empty_password_on_sign_in(accounts):
    with pytest.raises(AuthError, match="password"):
        accounts.sign_in("a@x.org", "")


def test_unknown_backend_failure_is_generic():
    class Broken:
        def create_user(self, email, password):
            raise AuthFailed("other", "500 internal")

    with pytest.raises(AuthError, match="try again"):
        Accounts(Broken()).sign_up("a@x.org", "password1")


def test_unavailable_backend_propagates():
    class Down:
        def verify_password(self, email, password):
            raise BackendUnavailable("paused")

    with pytest.raises(BackendUnavailable):
        Accounts(Down()).sign_in("a@x.org", "password1")
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_accounts.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.accounts'`

- [ ] **Step 3: Implement `medrag/accounts.py`**

```python
"""Sign-up and log-in. A user is identified by the id the auth backend issues; the app keeps
that id in the Streamlit session and passes it to every data call."""
import re
from dataclasses import dataclass

from medrag.cloud.base import Auth, AuthFailed

MIN_PASSWORD = 8
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MESSAGES = {
    "exists": "An account with this email already exists. Log in instead.",
    "invalid_credentials": "Wrong email or password.",
    "weak_password": f"Choose a stronger password (at least {MIN_PASSWORD} characters).",
    "other": "Sign-in failed. Please try again in a moment.",
}


class AuthError(ValueError):
    """A sign-up or log-in problem; str(error) is safe to show to the user."""


@dataclass(frozen=True)
class User:
    id: str
    email: str


def _clean_email(email: str) -> str:
    cleaned = email.strip().lower()
    if not _EMAIL.match(cleaned):
        raise AuthError("Enter a valid email address.")
    return cleaned


class Accounts:
    def __init__(self, auth: Auth):
        self.auth = auth

    def sign_up(self, email: str, password: str) -> User:
        email = _clean_email(email)
        if len(password) < MIN_PASSWORD:
            raise AuthError(_MESSAGES["weak_password"])
        try:
            return User(self.auth.create_user(email, password), email)
        except AuthFailed as exc:
            raise AuthError(_MESSAGES.get(exc.code, _MESSAGES["other"])) from exc

    def sign_in(self, email: str, password: str) -> User:
        email = _clean_email(email)
        if not password:
            raise AuthError("Enter your password.")
        try:
            return User(self.auth.verify_password(email, password), email)
        except AuthFailed as exc:
            raise AuthError(_MESSAGES.get(exc.code, _MESSAGES["other"])) from exc
```

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm dev pytest tests/test_accounts.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medrag/accounts.py tests/test_accounts.py
git commit -m "feat(accounts): email/password sign-up and log-in with friendly errors"
```

---

### Task 5: Encrypted Groq keys

**Files:**
- Create: `medrag/keys.py`
- Test: `tests/test_keys.py`

**Interfaces:**
- Consumes: `Database.get_user_key / set_user_key / delete_user_key`.
- Produces: `KeyRejected(ValueError)`; `mask(key: str) -> str` (`"gsk_…abcd"`); `check_groq_key(key: str) -> bool` (network; `False` if Groq says the key is invalid); `KeyVault(db, fernet_key: str, checker=check_groq_key)` with `save(user_id, key)`, `get(user_id) -> str | None`, `has_key(user_id) -> bool`, `masked(user_id) -> str | None`, `remove(user_id)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_keys.py`:

```python
import pytest
from cryptography.fernet import Fernet

from medrag.cloud.memory import MemoryDatabase
from medrag.keys import KeyRejected, KeyVault, mask

GOOD = "gsk_" + "A1b2C3d4" * 6


@pytest.fixture
def db():
    return MemoryDatabase()


def vault(db, checker=lambda key: True, secret=None):
    return KeyVault(db, secret or Fernet.generate_key().decode(), checker=checker)


def test_save_encrypts_and_get_decrypts(db):
    v = vault(db)
    v.save("alice", f"  {GOOD}\n")
    assert v.get("alice") == GOOD
    assert GOOD not in db.get_user_key("alice")
    assert v.has_key("alice") and not v.has_key("bob")


def test_masked_shows_only_prefix_and_last_four(db):
    v = vault(db)
    v.save("alice", GOOD)
    assert v.masked("alice") == mask(GOOD) == f"gsk_…{GOOD[-4:]}"
    assert v.masked("bob") is None


def test_rejects_malformed_key_without_calling_groq(db):
    calls = []
    v = vault(db, checker=lambda k: calls.append(k) or True)
    with pytest.raises(KeyRejected, match="gsk_"):
        v.save("alice", "sk-not-groq")
    assert calls == [] and not v.has_key("alice")


def test_rejects_key_groq_refuses(db):
    with pytest.raises(KeyRejected, match="rejected"):
        vault(db, checker=lambda k: False).save("alice", GOOD)


def test_remove(db):
    v = vault(db)
    v.save("alice", GOOD)
    v.remove("alice")
    assert v.get("alice") is None


def test_wrong_secret_reads_as_no_key(db):
    vault(db).save("alice", GOOD)
    assert vault(db).get("alice") is None  # a different Fernet secret cannot decrypt
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_keys.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.keys'`

- [ ] **Step 3: Implement `medrag/keys.py`**

```python
"""Users' own Groq keys: checked with Groq once, stored Fernet-encrypted, shown only masked."""
import re
from typing import Callable

from cryptography.fernet import Fernet, InvalidToken

from medrag.cloud.base import Database

_KEY_FORMAT = re.compile(r"^gsk_[A-Za-z0-9]{20,}$")


class KeyRejected(ValueError):
    """str(error) is safe to show to the user."""


def mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}"


def check_groq_key(key: str) -> bool:
    """Lists models (free, no tokens). False if Groq rejects the key."""
    from groq import AuthenticationError, Groq

    try:
        Groq(api_key=key, max_retries=0, timeout=15.0).models.list()
    except AuthenticationError:
        return False
    except Exception as exc:
        raise KeyRejected("Couldn't reach Groq to check the key. Try again in a minute.") from exc
    return True


class KeyVault:
    def __init__(self, db: Database, fernet_key: str, checker: Callable[[str], bool] = check_groq_key):
        self._db = db
        self._fernet = Fernet(fernet_key.encode())
        self._check = checker

    def save(self, user_id: str, key: str) -> None:
        key = key.strip()
        if not _KEY_FORMAT.match(key):
            raise KeyRejected("That doesn't look like a Groq API key. It should start with gsk_.")
        if not self._check(key):
            raise KeyRejected("Groq rejected this key. Check it and try again.")
        self._db.set_user_key(user_id, self._fernet.encrypt(key.encode()).decode())

    def get(self, user_id: str) -> str | None:
        token = self._db.get_user_key(user_id)
        if not token:
            return None
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except InvalidToken:
            return None

    def has_key(self, user_id: str) -> bool:
        return self.get(user_id) is not None

    def masked(self, user_id: str) -> str | None:
        key = self.get(user_id)
        return mask(key) if key else None

    def remove(self, user_id: str) -> None:
        self._db.delete_user_key(user_id)
```

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm dev pytest tests/test_keys.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medrag/keys.py tests/test_keys.py
git commit -m "feat(keys): Fernet-encrypted user Groq keys with validation and masking"
```

---

### Task 6: Shared-key quota

**Files:**
- Create: `medrag/quota.py`
- Test: `tests/test_quota.py`

**Interfaces:**
- Consumes: `Database.usage / increment_usage`; `KeyVault.get`.
- Produces: `KeyChoice(source: str, api_key: str)` (source `"shared" | "own" | "none"`); `utc_today() -> date`; `counts_toward_cap(answer: Answer) -> bool`; `Quota(db, vault, shared_key: str, daily_cap: int, today=utc_today)` with `used_today(user_id) -> int`, `own_key(user_id) -> str | None`, `choose(user_id) -> KeyChoice`, `record(user_id) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quota.py`:

```python
from datetime import date

import pytest
from cryptography.fernet import Fernet

from medrag.cloud.memory import MemoryDatabase
from medrag.keys import KeyVault
from medrag.quota import KeyChoice, Quota, counts_toward_cap
from medrag.types import Answer

OWN = "gsk_" + "Z9" * 12


class Clock:
    def __init__(self):
        self.day = date(2026, 10, 15)

    def __call__(self):
        return self.day


@pytest.fixture
def parts():
    db = MemoryDatabase()
    vault = KeyVault(db, Fernet.generate_key().decode(), checker=lambda k: True)
    clock = Clock()
    return db, vault, clock, Quota(db, vault, "gsk_shared", daily_cap=2, today=clock)


def test_shared_key_until_cap_then_none(parts):
    _, _, _, q = parts
    assert q.choose("alice") == KeyChoice("shared", "gsk_shared")
    q.record("alice")
    q.record("alice")
    assert q.used_today("alice") == 2
    assert q.choose("alice") == KeyChoice("none", "")


def test_own_key_after_cap(parts):
    _, vault, _, q = parts
    vault.save("alice", OWN)
    q.record("alice")
    q.record("alice")
    assert q.choose("alice") == KeyChoice("own", OWN)
    assert q.own_key("alice") == OWN


def test_usage_is_per_user(parts):
    _, _, _, q = parts
    q.record("alice")
    q.record("alice")
    assert q.choose("bob").source == "shared"
    assert q.used_today("bob") == 0


def test_cap_resets_at_utc_midnight(parts):
    _, _, clock, q = parts
    q.record("alice")
    q.record("alice")
    clock.day = date(2026, 10, 16)
    assert q.used_today("alice") == 0
    assert q.choose("alice").source == "shared"


def test_no_shared_key_configured_uses_own_key_or_none():
    db = MemoryDatabase()
    vault = KeyVault(db, Fernet.generate_key().decode(), checker=lambda k: True)
    q = Quota(db, vault, "", daily_cap=10)
    assert q.choose("alice").source == "none"
    vault.save("alice", OWN)
    assert q.choose("alice") == KeyChoice("own", OWN)


@pytest.mark.parametrize("refused, reason, counts", [
    (False, None, True),
    (True, "not_in_context", True),
    (True, "low_relevance", False),
    (True, "invalid_input", False),
    (True, "llm_error", False),
    (True, "rate_limited", False),
    (True, "key_rejected", False),
])
def test_only_generated_answers_count(refused, reason, counts):
    assert counts_toward_cap(Answer(text="x", refused=refused, refusal_reason=reason)) is counts
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_quota.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.quota'`

- [ ] **Step 3: Implement `medrag/quota.py`**

```python
"""Which Groq key answers a question: the shared app key while the user is under today's cap
(UTC day), then the user's own key. Only answers the model actually generated count."""
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable

from medrag.cloud.base import Database
from medrag.keys import KeyVault
from medrag.types import Answer


@dataclass(frozen=True)
class KeyChoice:
    source: str  # "shared" | "own" | "none"
    api_key: str


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def counts_toward_cap(answer: Answer) -> bool:
    return not answer.refused or answer.refusal_reason == "not_in_context"


class Quota:
    def __init__(
        self, db: Database, vault: KeyVault, shared_key: str, daily_cap: int,
        today: Callable[[], date] = utc_today,
    ):
        self._db = db
        self._vault = vault
        self.shared_key = shared_key
        self.daily_cap = daily_cap
        self._today = today

    def used_today(self, user_id: str) -> int:
        return self._db.usage(user_id, self._today())

    def own_key(self, user_id: str) -> str | None:
        return self._vault.get(user_id)

    def choose(self, user_id: str) -> KeyChoice:
        if self.shared_key and self.used_today(user_id) < self.daily_cap:
            return KeyChoice("shared", self.shared_key)
        own = self.own_key(user_id)
        return KeyChoice("own", own) if own else KeyChoice("none", "")

    def record(self, user_id: str) -> int:
        return self._db.increment_usage(user_id, self._today())
```

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm dev pytest tests/test_quota.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medrag/quota.py tests/test_quota.py
git commit -m "feat(quota): per-user daily cap on the shared Groq key with own-key fallback"
```

---

### Task 7: PDFs from bytes, shared chunk building, upload pre-checks

**Files:**
- Modify: `medrag/ingest/pdf.py`, `medrag/ingest/build.py`
- Create: `medrag/ingest/precheck.py`
- Create: `tests/fakes_cloud.py`
- Test: `tests/test_precheck.py`, `tests/test_ingest.py` (add one test)

**Interfaces:**
- Produces:
  - `read_pdf(source: Path | bytes) -> RawDocument`, `read_font_lines(source: Path | bytes)` (paths keep working).
  - `chunks_for_pdf(source: Path | bytes, doc_id: str, title: str, count_tokens, max_tokens: int, overlap_tokens: int) -> tuple[list[Chunk], IngestStats]` in `medrag.ingest.build`; `ingest_pdf` uses it.
  - `medrag.ingest.precheck`: `PdfRejected(ValueError)`, `PdfCheck(pages: int, text_pages: int)`, `check_pdf(data: bytes, max_bytes: int, max_pages: int) -> PdfCheck`, constants `NOT_A_PDF_MESSAGE`, `SCANNED_MESSAGE`.
  - `tests/fakes_cloud.py`: `BODIES` (4 page texts), `make_pdf_bytes(bodies, blank=(), watermark=False) -> bytes`.

- [ ] **Step 1: Create the test PDF helper**

Create `tests/fakes_cloud.py`:

```python
"""Helpers for multi-user tests: small generated PDFs (more helpers are added in Task 11)."""
import pymupdf

BODIES = [
    "Tinnitus is ringing in the ear.\nIt may be subjective or objective.\nMaskers help some patients.\nRetraining aims at habituation.",
    "Otosclerosis causes conductive deafness.\nStapedectomy is the usual operation.\nHearing aids are an option.\nFluoride may slow it.",
    "The nasal septum has three parts.\nThese are columellar, membranous\nand septum proper.\nDeviation causes obstruction.",
    "Acute sinusitis follows a cold.\nMaxillary sinus is most affected.\nTreatment uses antibiotics.\nDecongestants aid drainage.",
]


def make_pdf_bytes(bodies: list[str], blank=(), watermark: bool = False) -> bytes:
    """One page per body with a running header and a printed page number from 101.
    Pages listed in `blank` get no body: only the header, number and optional watermark,
    which is what an image-only (scanned) page looks like to text extraction."""
    doc = pymupdf.open()
    for n, body in enumerate(bodies):
        page = doc.new_page()
        page.insert_text((72, 60), "Test Book of ENT")
        if n not in blank:
            page.insert_text((72, 120), body)
        page.insert_text((300, 800), str(n + 101))
        if watermark:
            page.insert_text((72, 820), "mebooksfree.com")
    data = doc.tobytes()
    doc.close()
    return data
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_precheck.py`:

```python
import pymupdf
import pytest

from medrag.ingest.precheck import NOT_A_PDF_MESSAGE, SCANNED_MESSAGE, PdfRejected, check_pdf
from tests.fakes_cloud import BODIES, make_pdf_bytes

MB = 1024 * 1024
TEN = (BODIES * 3)[:10]


def check(data, max_bytes=50 * MB, max_pages=1000):
    return check_pdf(data, max_bytes=max_bytes, max_pages=max_pages)


def test_text_pdf_passes():
    result = check(make_pdf_bytes(BODIES))
    assert (result.pages, result.text_pages) == (4, 4)


def test_a_few_blank_pages_are_fine():
    assert check(make_pdf_bytes(TEN, blank={3})).text_pages == 9


def test_scanned_pdf_is_rejected():
    with pytest.raises(PdfRejected) as e:
        check(make_pdf_bytes(TEN, blank={1, 5}))  # 20% without text
    assert str(e.value) == SCANNED_MESSAGE


def test_watermarked_image_only_pages_count_as_scanned():
    with pytest.raises(PdfRejected, match="no text layer"):
        check(make_pdf_bytes(TEN, blank=set(range(10)), watermark=True))


def test_not_a_pdf():
    with pytest.raises(PdfRejected) as e:
        check(b"hello, this is not a pdf")
    assert str(e.value) == NOT_A_PDF_MESSAGE


def test_encrypted_pdf_is_rejected():
    doc = pymupdf.open(stream=make_pdf_bytes(BODIES), filetype="pdf")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(PdfRejected, match="couldn't be opened"):
        check(data)


def test_too_large():
    data = make_pdf_bytes(BODIES)
    with pytest.raises(PdfRejected, match="limit is"):
        check(data, max_bytes=len(data) - 1)


def test_too_many_pages():
    with pytest.raises(PdfRejected, match="4 pages; the limit is 3"):
        check(make_pdf_bytes(BODIES), max_pages=3)
```

Append to `tests/test_ingest.py`:

```python
from medrag.ingest.build import chunks_for_pdf
from tests.fakes_cloud import make_pdf_bytes


def test_chunks_for_pdf_reads_bytes_like_a_file(tmp_path):
    data = make_pdf_bytes([EAR_1, EAR_2, NOSE_1, NOSE_2])
    path = tmp_path / "book.pdf"
    path.write_bytes(data)
    from_bytes, stats = chunks_for_pdf(data, "b1", "Book", FakeEmbedder().count_tokens, 50, 10)
    from_path, _ = chunks_for_pdf(path, "b1", "Book", FakeEmbedder().count_tokens, 50, 10)
    assert from_bytes == from_path and stats.chunks == len(from_bytes) > 0
    assert stats.pages_total == 4 and from_bytes[0].doc_id == "b1"
```

- [ ] **Step 3: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_precheck.py tests/test_ingest.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.ingest.precheck'` and `ImportError: cannot import name 'chunks_for_pdf'`

- [ ] **Step 4: Read PDFs from bytes**

In `medrag/ingest/pdf.py` add, after `RawDocument`:

```python
def _open(source: "Path | bytes"):
    import pymupdf

    if isinstance(source, (bytes, bytearray)):
        return pymupdf.open(stream=bytes(source), filetype="pdf")
    return pymupdf.open(str(source))
```

and replace `with pymupdf.open(str(path)) as doc:` in both `read_pdf` and `read_font_lines` with `with _open(source) as doc:`, renaming each function's parameter from `path: Path` to `source: Path | bytes` and deleting their now-unused `import pymupdf` lines.

- [ ] **Step 5: Extract `chunks_for_pdf`**

In `medrag/ingest/build.py`, add above `ingest_pdf`:

```python
def chunks_for_pdf(
    source: Path | bytes,
    doc_id: str,
    title: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[list[Chunk], IngestStats]:
    raw = read_pdf(source)
    n_pages = len(raw.pages_lines)
    sections = (
        sections_from_toc(raw.toc, n_pages) if raw.toc else sections_from_fonts(read_font_lines(source))
    )
    chunks, pages_indexed, pages_without_text = build_chunks(
        doc_id, title, raw.pages_lines, raw.pdf_labels, sections, count_tokens, max_tokens, overlap_tokens,
    )
    return chunks, IngestStats(
        doc_id=doc_id, title=title, pages_total=n_pages, pages_indexed=pages_indexed,
        pages_without_text=pages_without_text, chunks=len(chunks),
    )
```

and replace the body of `ingest_pdf` with:

```python
    path = Path(path)
    doc_id = slugify(path.stem)
    title = title or path.stem.replace("_", " ").replace("-", " ").title()
    chunks, stats = chunks_for_pdf(path, doc_id, title, embedder.count_tokens, max_tokens, overlap_tokens)
    store.delete_document(doc_id)
    if chunks:
        store.add(chunks, embedder.encode([context_text(c) for c in chunks]))
    return stats
```

- [ ] **Step 6: Implement `medrag/ingest/precheck.py`**

```python
"""Checks an upload before anything is stored: it opens as a PDF, is within the size and
page limits, and has a text layer. Scanned books are rejected rather than OCR'd."""
from dataclasses import dataclass

from medrag.ingest.clean import find_repeating_margin_lines, lines_to_text, strip_margin_lines

NOT_A_PDF_MESSAGE = "This file couldn't be opened as a PDF."
SCANNED_MESSAGE = "This PDF has no text layer (it looks scanned). Upload a text-based PDF."
MIN_PAGE_CHARS = 100  # same rule ingestion uses for "a page without text"
MAX_NO_TEXT_SHARE = 0.2


class PdfRejected(ValueError):
    """str(error) is safe to show to the user."""


@dataclass(frozen=True)
class PdfCheck:
    pages: int
    text_pages: int


def check_pdf(data: bytes, max_bytes: int, max_pages: int) -> PdfCheck:
    import pymupdf

    if len(data) > max_bytes:
        mb = 1024 * 1024
        raise PdfRejected(f"This PDF is {len(data) / mb:.1f} MB; the limit is {max_bytes // mb} MB.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PdfRejected(NOT_A_PDF_MESSAGE) from exc
    with doc:
        if doc.needs_pass or doc.page_count == 0:
            raise PdfRejected(NOT_A_PDF_MESSAGE)
        if doc.page_count > max_pages:
            raise PdfRejected(f"This PDF has {doc.page_count} pages; the limit is {max_pages}.")
        try:
            pages_lines = [page.get_text("text").splitlines() for page in doc]
        except Exception as exc:
            raise PdfRejected(NOT_A_PDF_MESSAGE) from exc
    repeating = find_repeating_margin_lines(pages_lines)
    text_pages = sum(
        1 for lines in pages_lines if len(lines_to_text(strip_margin_lines(lines, repeating))) >= MIN_PAGE_CHARS
    )
    pages = len(pages_lines)
    if pages - text_pages >= MAX_NO_TEXT_SHARE * pages:
        raise PdfRejected(SCANNED_MESSAGE)
    return PdfCheck(pages=pages, text_pages=text_pages)
```

- [ ] **Step 7: Run the tests**

Run: `docker compose run --rm dev pytest -q`
Expected: all pass. If `test_a_few_blank_pages_are_fine` fails because a body page measures under 100 characters, print `len(lines_to_text(...))` per page in a scratch script and lengthen that `BODIES` entry in `tests/fakes_cloud.py`; do not lower `MIN_PAGE_CHARS`.

- [ ] **Step 8: Commit**

```bash
git add medrag/ingest/pdf.py medrag/ingest/build.py medrag/ingest/precheck.py tests/fakes_cloud.py tests/test_precheck.py tests/test_ingest.py
git commit -m "feat(ingest): read PDFs from bytes, shared chunks_for_pdf, upload pre-checks (scanned/corrupt/limits)"
```

---

### Task 8: Chunk rows and the in-memory chunk store

**Files:**
- Create: `medrag/store_pg.py`
- Test: `tests/test_store_pg.py`

**Interfaces:**
- Consumes: `Chunk` from `medrag.types`; row keys from Task 3.
- Produces: `chunk_row(chunk: Chunk, embedding: list[float], user_id: str) -> dict` (`book_id = chunk.doc_id`, embedding rounded to 6 decimals); `row_chunk(row: dict, doc_title: str) -> Chunk`; `InMemoryChunkStore(chunks, embeddings)` with `from_rows(rows, doc_title)`, `count()`, `all_chunks()`, `documents() -> dict[str, str]`, `query(embedding, n, doc_ids=None) -> list[tuple[Chunk, float]]` (cosine, best first).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_store_pg.py`:

```python
from medrag.retrieval.bm25 import BM25Index
from medrag.retrieval.retriever import HybridRetriever
from medrag.store_pg import InMemoryChunkStore, chunk_row, row_chunk
from tests.fakes import FakeEmbedder, OverlapReranker, make_chunk

TEXTS = ["tinnitus is ringing in the ear", "sinusitis is inflammation of the sinuses", "the nasal septum has three parts"]


def store_of(chunks):
    return InMemoryChunkStore(chunks, FakeEmbedder().encode([c.text for c in chunks]))


def test_query_ranks_by_cosine():
    chunks = [make_chunk(i, t) for i, t in enumerate(TEXTS)]
    hits = store_of(chunks).query(FakeEmbedder().encode(["ringing ear tinnitus"])[0], n=2)
    assert len(hits) == 2 and hits[0][0] == chunks[0] and hits[0][1] >= hits[1][1]


def test_query_filters_by_document_and_caps_n():
    chunks = [make_chunk(0, TEXTS[0], doc="a"), make_chunk(1, TEXTS[1], doc="b")]
    s = store_of(chunks)
    q = FakeEmbedder().encode(["sinus"])[0]
    assert [c.doc_id for c, _ in s.query(q, n=10, doc_ids=["b"])] == ["b"]
    assert s.query(q, n=10, doc_ids=["zzz"]) == []
    assert len(s.query(q, n=10)) == 2


def test_empty_store():
    s = InMemoryChunkStore([], [])
    assert s.count() == 0 and s.all_chunks() == [] and s.documents() == {}
    assert s.query([0.1, 0.2], n=5) == []


def test_rows_roundtrip_with_title_and_user():
    c = make_chunk(3, TEXTS[0], doc="book-1", page=7, section="Ear > Tinnitus")
    c = type(c)(**{**c.__dict__, "topic": "Tinnitus"})
    r = chunk_row(c, [0.12345678, 1.0], user_id="alice")
    assert r["book_id"] == "book-1" and r["user_id"] == "alice" and r["embedding"] == [0.123457, 1.0]
    back = row_chunk(r, doc_title=c.doc_title)
    assert back == c
    s = InMemoryChunkStore.from_rows([r], "Book One")
    assert s.documents() == {"book-1": "Book One"} and s.count() == 1


def test_plugs_into_the_hybrid_retriever():
    chunks = [make_chunk(i, t) for i, t in enumerate(TEXTS)]
    retriever = HybridRetriever(store_of(chunks), FakeEmbedder(), BM25Index(chunks), OverlapReranker(), top_n=2)
    results = retriever.retrieve("what is tinnitus")
    assert results[0].chunk == chunks[0]
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_store_pg.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.store_pg'`

- [ ] **Step 3: Implement `medrag/store_pg.py`**

```python
"""Chunks as Postgres rows, and an in-memory vector store over one book's chunks. A book is
loaded from the database once and searched here; dense search is brute-force cosine, which
takes milliseconds for a few thousand chunks."""
import numpy as np

from medrag.types import Chunk


def chunk_row(chunk: Chunk, embedding: list[float], user_id: str) -> dict:
    return {
        "id": chunk.id,
        "book_id": chunk.doc_id,
        "user_id": user_id,
        "text": chunk.text,
        "page_index": chunk.page_index,
        "page_label": chunk.page_label,
        "section": chunk.section,
        "topic": chunk.topic,
        "embedding": [round(float(x), 6) for x in embedding],
    }


def row_chunk(row: dict, doc_title: str) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["book_id"],
        doc_title=doc_title,
        text=row["text"],
        page_index=int(row["page_index"]),
        page_label=str(row["page_label"]),
        section=row.get("section") or "",
        topic=row.get("topic") or "",
    )


class InMemoryChunkStore:
    """Read-only stand-in for ChunkStore with the methods retrieval uses."""

    def __init__(self, chunks: list[Chunk], embeddings: list[list[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings differ in length")
        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    @classmethod
    def from_rows(cls, rows: list[dict], doc_title: str) -> "InMemoryChunkStore":
        return cls([row_chunk(r, doc_title) for r in rows], [r["embedding"] for r in rows])

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def documents(self) -> dict[str, str]:
        return {c.doc_id: c.doc_title for c in self._chunks}

    def query(
        self, embedding: list[float], n: int, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        if not self._chunks or n <= 0:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._matrix @ q
        hits: list[tuple[Chunk, float]] = []
        for i in np.argsort(-scores, kind="stable"):
            chunk = self._chunks[i]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            hits.append((chunk, float(scores[i])))
            if len(hits) == n:
                break
        return hits
```

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm dev pytest tests/test_store_pg.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medrag/store_pg.py tests/test_store_pg.py
git commit -m "feat(store): chunk rows and an in-memory per-book vector store"
```

---

### Task 9: Library

**Files:**
- Create: `medrag/library.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `Backend`, `Book`, `utc_now` (Task 3); `CloudSettings` (Task 1); `check_pdf`, `PdfRejected`, `PdfCheck` (Task 7); `chunk_row` (Task 8).
- Produces: `UploadRejected(ValueError)`; `INTERRUPTED_MESSAGE`; `clean_title(title: str, filename: str) -> str`; `Library(backend, cloud, new_id=uuid4-str)` with attribute `on_delete: list[Callable[[str], None]]` and methods `list(user_id) -> list[Book]`, `get(user_id, book_id) -> Book | None`, `check_upload(user_id, data: bytes) -> PdfCheck`, `add(user_id, title, filename, data) -> Book` (status `processing`, progress `"Queued"`), `set_progress(user_id, book_id, text)`, `insert_chunks(user_id, chunks, embeddings)`, `mark_ready(user_id, book_id, pages, chunks)`, `mark_failed(user_id, book_id, reason)` (also deletes chunks), `delete(user_id, book_id) -> bool` (row + chunks + PDF, then calls every `on_delete(book_id)`), `fail_interrupted() -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_library.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_library.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.library'`

- [ ] **Step 3: Implement `medrag/library.py`**

```python
"""A user's private books: upload limits, storage, status and deletion. Every method takes
user_id; the store filters on it, so one user can never see or touch another's books."""
import uuid
from pathlib import PurePath
from typing import Callable

from medrag.cloud.base import Backend, Book
from medrag.config import CloudSettings
from medrag.ingest.precheck import PdfCheck, PdfRejected, check_pdf
from medrag.store_pg import chunk_row
from medrag.types import Chunk

MB = 1024 * 1024
MAX_TITLE = 200
INTERRUPTED_MESSAGE = "Indexing was interrupted because the app restarted. Delete the book and upload it again."


class UploadRejected(ValueError):
    """str(error) is safe to show to the user."""


def clean_title(title: str, filename: str) -> str:
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
            id=book_id, user_id=user_id, title=clean_title(title, filename), filename=filename[:255],
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

    def mark_ready(self, user_id: str, book_id: str, pages: int, chunks: int) -> None:
        self.db.update_book(user_id, book_id, {"status": "ready", "pages": pages, "chunks": chunks, "progress": "", "error": ""})

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
```

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm dev pytest tests/test_library.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add medrag/library.py tests/test_library.py
git commit -m "feat(library): per-user books with upload limits, status changes and deletion"
```

---

### Task 10: Background ingestion worker

**Files:**
- Create: `medrag/ingest/job.py`
- Test: `tests/test_ingest_job.py`

**Interfaces:**
- Consumes: `Library` (Task 9), `Storage` (Task 3), `chunks_for_pdf` (Task 7), `Embedder`, `Settings`, `context_text`.
- Produces: `IngestFailed(Exception)`; `IngestWorker(library, storage, embedder, settings, threaded=True, batch=128)` with `submit(user_id, book_id)` (runs inline when `threaded=False`), `run_one(user_id, book_id)` (never raises), `wait_idle(timeout=30.0) -> bool`. Messages `NO_TEXT_MESSAGE`, `UNEXPECTED_MESSAGE`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingest_job.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_ingest_job.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.ingest.job'`

- [ ] **Step 3: Implement `medrag/ingest/job.py`**

```python
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
            self.library.set_progress(user_id, book.id, f"Indexing passages {start}/{len(chunks)}")
            part = slice(start, start + self.batch)
            self.library.insert_chunks(user_id, chunks[part], self.embedder.encode(texts[part]))
        self.library.mark_ready(user_id, book.id, pages=stats.pages_total, chunks=len(chunks))
```

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm dev pytest tests/test_ingest_job.py -q`
Expected: PASS. In the deletion test, the first `insert_chunks` raises `BackendError` because the book no longer exists; `mark_failed` then finds no row and does nothing.

- [ ] **Step 5: Commit**

```bash
git add medrag/ingest/job.py tests/test_ingest_job.py
git commit -m "feat(ingest): single-thread background worker with progress and safe failure"
```

---

### Task 11: Chat service and wiring

**Files:**
- Create: `medrag/chat.py`, `medrag/services.py`
- Modify: `tests/fakes_cloud.py` (add helpers)
- Test: `tests/test_chat.py`, `tests/test_services.py`

**Interfaces:**
- Consumes: everything from Tasks 3–10; `MedRAG`, `Generator`, `HybridRetriever`, `BM25Index`.
- Produces:
  - `medrag.chat`: `ChatService(library, db, models, quota, settings, cache_size=4)` with `ask(user_id, book_id, question, history=None) -> Answer`, `evict(book_id)`, `cached_ids() -> list[str]`; messages `BOOK_UNAVAILABLE_MESSAGE`, `QUOTA_MESSAGE` (format with `cap`), `SHARED_EXHAUSTED_MESSAGE`, `OWN_KEY_REJECTED_MESSAGE`, `OWN_KEY_LIMIT_MESSAGE`. Extra refusal reasons: `"book_unavailable"`, `"quota"`.
  - `medrag.services`: `Models(embedder, reranker, gate, llm_factory: Callable[[str], LLMClient])`; `Services(settings, cloud, backend, accounts, library, vault, quota, worker, chat)`; `assemble(backend, models, settings, cloud, key_checker=check_groq_key, threaded=True) -> Services` (also runs `library.fail_interrupted()` and wires `chat.evict` into `library.on_delete`); `upload_book(services, user_id, title, filename, data, confirmed: bool) -> Book` (raises `UploadRejected`); `delete_book(services, user_id, book_id) -> bool`.
  - `tests/fakes_cloud.py`: `KeyedLLM`, `make_services(**overrides) -> tuple[Services, KeyedLLM]`, `ready_book(services, user_id, bodies=BODIES) -> Book`.

- [ ] **Step 1: Add the test helpers**

Append to `tests/fakes_cloud.py`:

```python
from cryptography.fernet import Fernet

from medrag.cloud.memory import memory_backend
from medrag.config import CloudSettings, Settings
from tests.fakes import FakeEmbedder, OverlapReranker

REPLY = "Tinnitus is ringing in the ear [1]."


class KeyedLLM:
    """LLM factory for tests: every client replies `reply`, except keys listed in `errors`,
    whose clients raise that error. `keys_used` records which key served each call."""

    def __init__(self, reply: str = REPLY, errors: dict | None = None):
        self.reply = reply
        self.errors = dict(errors or {})
        self.keys_used: list[str] = []

    def __call__(self, api_key: str):
        factory = self

        class Client:
            def complete(self, messages, temperature=0.0, max_tokens=1024):
                factory.keys_used.append(api_key)
                if api_key in factory.errors:
                    raise factory.errors[api_key]
                return factory.reply

        return Client()


def make_services(shared_key: str = "gsk_shared", cap: int = 2, llm: KeyedLLM | None = None, **cloud_overrides):
    from medrag.services import Models, assemble

    llm = llm or KeyedLLM()
    settings = Settings(groq_api_key=shared_key, rerank_threshold=0.045, top_n=3,
                        chunk_max_tokens=50, chunk_overlap_tokens=10)
    cloud = CloudSettings(backend="memory", fernet_key=Fernet.generate_key().decode(),
                          shared_daily_cap=cap, **cloud_overrides)
    models = Models(embedder=FakeEmbedder(), reranker=OverlapReranker(), gate=OverlapReranker(), llm_factory=llm)
    services = assemble(memory_backend(), models, settings, cloud, key_checker=lambda k: True, threaded=False)
    return services, llm


def ready_book(services, user_id: str, bodies=None, title: str = "ENT Notes"):
    from medrag.services import upload_book

    book = upload_book(services, user_id, title, "ent.pdf", make_pdf_bytes(bodies or BODIES), confirmed=True)
    return services.library.get(user_id, book.id)
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_services.py`:

```python
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
```

Create `tests/test_chat.py`:

```python
from medrag.chat import (
    BOOK_UNAVAILABLE_MESSAGE,
    OWN_KEY_LIMIT_MESSAGE,
    OWN_KEY_REJECTED_MESSAGE,
    QUOTA_MESSAGE,
    SHARED_EXHAUSTED_MESSAGE,
)
from medrag.generation.llm import LLMAuthError, LLMRateLimited
from medrag.services import delete_book
from tests.fakes_cloud import KeyedLLM, make_services, ready_book

OWN = "gsk_" + "Q7" * 12


def test_answers_from_the_book_on_the_shared_key_and_counts_it():
    s, llm = make_services()
    book = ready_book(s, "alice")
    answer = s.chat.ask("alice", book.id, "What is tinnitus?")
    assert not answer.refused and answer.citations
    assert answer.citations[0].chunk.doc_id == book.id and answer.citations[0].chunk.doc_title == "ENT Notes"
    assert llm.keys_used == ["gsk_shared"]
    assert s.quota.used_today("alice") == 1


def test_off_topic_question_is_refused_free():
    s, llm = make_services()
    book = ready_book(s, "alice")
    answer = s.chat.ask("alice", book.id, "Who won the football world cup?")
    assert answer.refusal_reason == "low_relevance"
    assert llm.keys_used == [] and s.quota.used_today("alice") == 0


def test_model_saying_not_found_still_counts():
    s, _ = make_services(llm=KeyedLLM(reply="NOT_FOUND"))
    book = ready_book(s, "alice")
    assert s.chat.ask("alice", book.id, "What is tinnitus?").refusal_reason == "not_in_context"
    assert s.quota.used_today("alice") == 1


def test_after_the_cap_without_own_key_the_user_is_told_to_add_one():
    s, llm = make_services(cap=1)
    book = ready_book(s, "alice")
    s.chat.ask("alice", book.id, "What is tinnitus?")
    answer = s.chat.ask("alice", book.id, "What is tinnitus?")
    assert answer.refusal_reason == "quota" and answer.text == QUOTA_MESSAGE.format(cap=1)
    assert llm.keys_used == ["gsk_shared"]


def test_after_the_cap_the_own_key_is_used_and_not_counted():
    s, llm = make_services(cap=1)
    s.vault.save("alice", OWN)
    book = ready_book(s, "alice")
    s.chat.ask("alice", book.id, "What is tinnitus?")
    assert not s.chat.ask("alice", book.id, "What is tinnitus?").refused
    assert llm.keys_used == ["gsk_shared", OWN]
    assert s.quota.used_today("alice") == 1


def test_shared_key_exhausted_falls_back_to_own_key():
    s, llm = make_services(llm=KeyedLLM(errors={"gsk_shared": LLMRateLimited("RateLimitError: tokens per day")}))
    s.vault.save("alice", OWN)
    book = ready_book(s, "alice")
    answer = s.chat.ask("alice", book.id, "What is tinnitus?")
    assert not answer.refused and llm.keys_used == ["gsk_shared", OWN]
    assert s.quota.used_today("alice") == 0


def test_shared_key_exhausted_without_own_key():
    s, _ = make_services(llm=KeyedLLM(errors={"gsk_shared": LLMRateLimited("RateLimitError")}))
    book = ready_book(s, "alice")
    answer = s.chat.ask("alice", book.id, "What is tinnitus?")
    assert answer.refused and answer.text == SHARED_EXHAUSTED_MESSAGE
    assert s.quota.used_today("alice") == 0


def test_own_key_problems_are_explained():
    s, _ = make_services(cap=0, llm=KeyedLLM(errors={OWN: LLMAuthError("AuthenticationError")}))
    s.vault.save("alice", OWN)
    book = ready_book(s, "alice")
    assert s.chat.ask("alice", book.id, "What is tinnitus?").text == OWN_KEY_REJECTED_MESSAGE
    s2, _ = make_services(cap=0, llm=KeyedLLM(errors={OWN: LLMRateLimited("RateLimitError")}))
    s2.vault.save("alice", OWN)
    book2 = ready_book(s2, "alice")
    assert s2.chat.ask("alice", book2.id, "What is tinnitus?").text == OWN_KEY_LIMIT_MESSAGE


def test_cannot_ask_about_someone_elses_book():
    s, llm = make_services()
    book = ready_book(s, "alice")
    answer = s.chat.ask("bob", book.id, "What is tinnitus?")
    assert answer.refusal_reason == "book_unavailable" and answer.text == BOOK_UNAVAILABLE_MESSAGE
    assert llm.keys_used == [] and s.quota.used_today("bob") == 0


def test_not_ready_or_deleted_books_are_unavailable():
    s, llm = make_services()
    from tests.fakes_cloud import BODIES, make_pdf_bytes

    pending = s.library.add("alice", "", "a.pdf", make_pdf_bytes(BODIES))
    assert s.chat.ask("alice", pending.id, "What is tinnitus?").refusal_reason == "book_unavailable"
    book = ready_book(s, "alice")
    delete_book(s, "alice", book.id)
    assert s.chat.ask("alice", book.id, "What is tinnitus?").refusal_reason == "book_unavailable"
    assert llm.keys_used == []


def test_retrievers_are_cached_per_book_with_lru_eviction():
    s, _ = make_services(cap=100, cached_books=2, max_books_per_user=3)
    loads = []
    original = s.backend.db.load_chunks
    s.chat.db.load_chunks = lambda u, b: (loads.append(b), original(u, b))[1]
    a, b, c = (ready_book(s, "alice", title=t) for t in ("A", "B", "C"))
    for book in (a, a, b, c, a):
        s.chat.ask("alice", book.id, "What is tinnitus?")
    assert loads == [a.id, b.id, c.id, a.id]
    assert s.chat.cached_ids() == [c.id, a.id]
```

- [ ] **Step 3: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_chat.py tests/test_services.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.services'`

- [ ] **Step 4: Implement `medrag/chat.py`**

```python
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
```

- [ ] **Step 5: Implement `medrag/services.py`**

```python
"""Builds the multi-user app's parts once per process and offers the two actions that span
several parts (upload, delete)."""
from dataclasses import dataclass
from typing import Callable

from cryptography.fernet import Fernet

from medrag.accounts import Accounts
from medrag.chat import ChatService
from medrag.cloud.base import Backend, Book
from medrag.config import CloudSettings, Settings
from medrag.embeddings import Embedder
from medrag.generation.llm import LLMClient
from medrag.ingest.job import IngestWorker
from medrag.keys import KeyVault, check_groq_key
from medrag.library import Library, UploadRejected
from medrag.quota import Quota
from medrag.retrieval.rerank import Reranker

CONFIRM_MESSAGE = "Please confirm this is a medical or health book and that you have the right to use it."


@dataclass
class Models:
    embedder: Embedder
    reranker: Reranker | None
    gate: Reranker | None
    llm_factory: Callable[[str], LLMClient]


@dataclass
class Services:
    settings: Settings
    cloud: CloudSettings
    backend: Backend
    accounts: Accounts
    library: Library
    vault: KeyVault
    quota: Quota
    worker: IngestWorker
    chat: ChatService


def assemble(
    backend: Backend, models: Models, settings: Settings, cloud: CloudSettings,
    key_checker: Callable[[str], bool] = check_groq_key, threaded: bool = True,
) -> Services:
    library = Library(backend, cloud)
    library.fail_interrupted()
    vault = KeyVault(backend.db, cloud.fernet_key or Fernet.generate_key().decode(), checker=key_checker)
    quota = Quota(backend.db, vault, settings.groq_api_key, cloud.shared_daily_cap)
    chat = ChatService(library, backend.db, models, quota, settings, cache_size=cloud.cached_books)
    library.on_delete.append(chat.evict)
    worker = IngestWorker(library, backend.storage, models.embedder, settings, threaded=threaded)
    return Services(settings, cloud, backend, Accounts(backend.auth), library, vault, quota, worker, chat)


def upload_book(services: Services, user_id: str, title: str, filename: str, data: bytes, confirmed: bool) -> Book:
    if not confirmed:
        raise UploadRejected(CONFIRM_MESSAGE)
    book = services.library.add(user_id, title, filename, data)
    services.worker.submit(user_id, book.id)
    return book


def delete_book(services: Services, user_id: str, book_id: str) -> bool:
    return services.library.delete(user_id, book_id)
```

(The memory backend gets a throwaway Fernet secret when none is set; Supabase mode always has one because `missing()` requires it, checked in Task 12.)

- [ ] **Step 6: Run the tests**

Run: `docker compose run --rm dev pytest -q`
Expected: all pass. `make_pdf_bytes`'s bodies all mention ear/nose topics; "What is tinnitus?" shares the term `tinnitus` with page 1, so the `OverlapReranker` gate scores 1.0 (above 0.045) and the football question scores 0.0.

- [ ] **Step 7: Commit**

```bash
git add medrag/chat.py medrag/services.py tests/fakes_cloud.py tests/test_chat.py tests/test_services.py
git commit -m "feat(chat): per-user book chat with key choice, fallback, quota counting and retriever cache"
```

---

### Task 12: Supabase backend, schema and production wiring

**Files:**
- Create: `medrag/cloud/supabase.py`, `supabase/schema.sql`
- Modify: `medrag/services.py` (add `make_backend`, `load_models`, `build_services`)
- Test: `tests/test_cloud_supabase.py` (unit), `tests/test_live_supabase.py` (`slow`)

**Interfaces:**
- Consumes: contracts from Task 3; `CloudSettings` (Task 1); `assemble` (Task 11).
- Produces: `supabase_backend(url, service_key, anon_key) -> Backend`; `auth_failure(exc) -> AuthFailed`; `guard` decorator mapping transport errors to `BackendUnavailable` and other store errors to `BackendError`; `SupabaseAuth.delete_user(user_id)` (tests only). In `medrag.services`: `make_backend(cloud) -> Backend`, `load_models(settings) -> Models`, `build_services(settings, cloud) -> Services` (raises `ValueError("Missing settings: ...")` when `cloud.missing()`).

- [ ] **Step 1: Write the schema**

Create `supabase/schema.sql`:

```sql
-- MedRAG multi-user schema. Run once in the Supabase SQL editor (safe to re-run).
-- The app talks to Postgres with the service-role key (server-side only), which bypasses RLS;
-- the app filters every query by user_id. RLS below is defence in depth for any future
-- anon/authenticated access: users may read only their own rows and write nothing directly.

create table if not exists public.books (
  id uuid primary key,
  user_id uuid not null references auth.users (id) on delete cascade,
  title text not null check (char_length(title) between 1 and 200),
  filename text not null,
  storage_path text not null,
  size_bytes bigint not null check (size_bytes >= 0),
  pages int not null default 0,
  chunks int not null default 0,
  status text not null check (status in ('processing', 'ready', 'failed')),
  progress text not null default '',
  error text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists books_user_id_idx on public.books (user_id);
create index if not exists books_status_idx on public.books (status);

create table if not exists public.chunks (
  id text primary key,
  book_id uuid not null references public.books (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  text text not null,
  page_index int not null,
  page_label text not null,
  section text not null default '',
  topic text not null default '',
  embedding real[] not null
);
create index if not exists chunks_book_id_idx on public.chunks (book_id);
create index if not exists chunks_user_id_idx on public.chunks (user_id);

create table if not exists public.usage (
  user_id uuid not null references auth.users (id) on delete cascade,
  day date not null,
  shared_key_questions int not null default 0 check (shared_key_questions >= 0),
  primary key (user_id, day)
);

create table if not exists public.user_keys (
  user_id uuid primary key references auth.users (id) on delete cascade,
  encrypted_groq_key text not null,
  created_at timestamptz not null default now()
);

alter table public.books enable row level security;
alter table public.chunks enable row level security;
alter table public.usage enable row level security;
alter table public.user_keys enable row level security;  -- no policy: nobody but the server

drop policy if exists "read own books" on public.books;
create policy "read own books" on public.books for select to authenticated using (user_id = auth.uid());
drop policy if exists "read own chunks" on public.chunks;
create policy "read own chunks" on public.chunks for select to authenticated using (user_id = auth.uid());
drop policy if exists "read own usage" on public.usage;
create policy "read own usage" on public.usage for select to authenticated using (user_id = auth.uid());

create or replace function public.increment_usage(p_user uuid, p_day date)
returns int language sql security invoker set search_path = '' as $$
  insert into public.usage (user_id, day, shared_key_questions) values (p_user, p_day, 1)
  on conflict (user_id, day) do update set shared_key_questions = public.usage.shared_key_questions + 1
  returning shared_key_questions;
$$;

create or replace function public.database_bytes()
returns bigint language sql stable security invoker set search_path = '' as $$
  select pg_database_size(current_database());
$$;

revoke all on function public.increment_usage(uuid, date) from public, anon, authenticated;
revoke all on function public.database_bytes() from public, anon, authenticated;
grant execute on function public.increment_usage(uuid, date) to service_role;
grant execute on function public.database_bytes() to service_role;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('books', 'books', false, 52428800, array['application/pdf'])
on conflict (id) do nothing;
```

- [ ] **Step 2: Write the failing unit tests**

Create `tests/test_cloud_supabase.py`:

```python
import httpx
import pytest

from medrag.cloud.base import AuthFailed, BackendError, BackendUnavailable
from medrag.cloud.supabase import auth_failure, guard


class FakeAuthApiError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize("exc, code", [
    (FakeAuthApiError("x", code="user_already_exists"), "exists"),
    (FakeAuthApiError("x", code="email_exists"), "exists"),
    (FakeAuthApiError("A user with this email address has already been registered"), "exists"),
    (FakeAuthApiError("x", code="invalid_credentials"), "invalid_credentials"),
    (FakeAuthApiError("Invalid login credentials"), "invalid_credentials"),
    (FakeAuthApiError("x", code="weak_password"), "weak_password"),
    (FakeAuthApiError("Password should be at least 6 characters"), "weak_password"),
    (FakeAuthApiError("Database error saving new user"), "other"),
])
def test_auth_failure_codes(exc, code):
    failure = auth_failure(exc)
    assert isinstance(failure, AuthFailed) and failure.code == code


def test_guard_maps_network_errors_to_unavailable():
    @guard
    def call():
        raise httpx.ConnectError("connection refused")

    with pytest.raises(BackendUnavailable):
        call()


def test_guard_maps_server_errors_to_unavailable():
    response = httpx.Response(503, request=httpx.Request("GET", "https://x"))

    @guard
    def call():
        raise httpx.HTTPStatusError("503", request=response.request, response=response)

    with pytest.raises(BackendUnavailable):
        call()


def test_guard_maps_other_errors_to_backend_error_and_keeps_ours():
    @guard
    def broken():
        raise RuntimeError("violates foreign key constraint")

    with pytest.raises(BackendError):
        broken()

    @guard
    def ours():
        raise BackendUnavailable("paused")

    with pytest.raises(BackendUnavailable):
        ours()


def test_build_services_reports_missing_settings():
    from medrag.config import CloudSettings, Settings
    from medrag.services import build_services

    with pytest.raises(ValueError, match="SUPABASE_ANON_KEY"):
        build_services(Settings(), CloudSettings(backend="supabase", supabase_url="https://x"))
```

- [ ] **Step 3: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_cloud_supabase.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'medrag.cloud.supabase'`

- [ ] **Step 4: Implement `medrag/cloud/supabase.py`**

```python
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
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


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
    def update_book(self, user_id: str, book_id: str, changes: dict) -> None:
        check_update_fields(changes)
        payload = {**changes, "updated_at": utc_now().isoformat()}
        self._books().update(payload).eq("user_id", user_id).eq("id", book_id).execute()

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
```

Verify the supabase-py import path against the pinned version before running: `docker compose run --rm dev python -c "from supabase import ClientOptions, create_client"`. If `ClientOptions` is not exported at top level in the pinned version, use `from supabase.lib.client_options import SyncClientOptions as ClientOptions` and keep the rest unchanged.

- [ ] **Step 5: Add production wiring to `medrag/services.py`**

Append:

```python
def make_backend(cloud: CloudSettings) -> Backend:
    if cloud.backend == "memory":
        from medrag.cloud.memory import memory_backend

        return memory_backend()
    if cloud.backend == "supabase":
        from medrag.cloud.supabase import supabase_backend

        return supabase_backend(cloud.supabase_url, cloud.supabase_service_key, cloud.supabase_anon_key)
    raise ValueError(f"MEDRAG_BACKEND must be 'memory' or 'supabase' for the multi-user app, not {cloud.backend!r}")


def load_models(settings: Settings) -> Models:
    from medrag.embeddings import SentenceTransformerEmbedder
    from medrag.generation.llm import GroqClient
    from medrag.retrieval.rerank import CrossEncoderReranker

    return Models(
        embedder=SentenceTransformerEmbedder(settings.embed_model),
        reranker=CrossEncoderReranker(settings.rerank_model),
        gate=CrossEncoderReranker(settings.gate_model) if settings.gate_model else None,
        llm_factory=lambda api_key: GroqClient(api_key, settings.llm_model),
    )


def build_services(settings: Settings, cloud: CloudSettings) -> Services:
    missing = cloud.missing()
    if missing:
        raise ValueError(f"Missing settings: {', '.join(missing)}")
    return assemble(make_backend(cloud), load_models(settings), settings, cloud)
```

- [ ] **Step 6: Run the unit tests**

Run: `docker compose run --rm dev pytest -q`
Expected: all pass.

- [ ] **Step 7: Write the live round-trip test**

Create `tests/test_live_supabase.py`:

```python
"""Round trip against a real Supabase project. Needs SUPABASE_URL, SUPABASE_ANON_KEY,
SUPABASE_SERVICE_KEY and MEDRAG_FERNET_KEY (from .env). Uses fake models and a fake LLM, so it
costs no Groq quota. Run: docker compose run --rm dev pytest -m slow tests/test_live_supabase.py"""
import uuid

import pytest

from medrag.config import CloudSettings, Settings
from medrag.services import Models, assemble, delete_book, make_backend, upload_book
from tests.fakes import FakeEmbedder, OverlapReranker
from tests.fakes_cloud import BODIES, KeyedLLM, make_pdf_bytes

pytestmark = pytest.mark.slow


@pytest.fixture
def live():
    cloud = CloudSettings.from_env()
    if cloud.backend != "supabase" or cloud.missing():
        pytest.skip("needs Supabase settings in .env")
    backend = make_backend(cloud)
    settings = Settings(groq_api_key="gsk_fake", chunk_max_tokens=50, chunk_overlap_tokens=10)
    models = Models(FakeEmbedder(), OverlapReranker(), OverlapReranker(), KeyedLLM())
    services = assemble(backend, models, settings, cloud, key_checker=lambda k: True, threaded=False)
    created: list[str] = []
    yield services, created
    for user_id in created:
        backend.auth.delete_user(user_id)


def test_sign_up_upload_ask_delete(live):
    services, created = live
    email = f"medrag-test-{uuid.uuid4().hex[:10]}@example.com"
    user = services.accounts.sign_up(email, "correct-horse-battery")
    created.append(user.id)
    assert services.accounts.sign_in(email, "correct-horse-battery").id == user.id

    book = upload_book(services, user.id, "Live test", "live.pdf", make_pdf_bytes(BODIES), confirmed=True)
    ready = services.library.get(user.id, book.id)
    assert ready.status == "ready" and ready.chunks > 0

    answer = services.chat.ask(user.id, book.id, "What is tinnitus?")
    assert not answer.refused and services.quota.used_today(user.id) == 1

    stranger = services.accounts.sign_up(f"medrag-test-{uuid.uuid4().hex[:10]}@example.com", "correct-horse-battery")
    created.append(stranger.id)
    assert services.library.get(stranger.id, book.id) is None
    assert services.chat.ask(stranger.id, book.id, "What is tinnitus?").refusal_reason == "book_unavailable"

    assert delete_book(services, user.id, book.id)
    assert services.library.get(user.id, book.id) is None
    assert services.backend.db.load_chunks(user.id, book.id) == []
```

This test runs only after the Supabase project exists (Task 15, `docs/deploy.md` steps 1–3). Until then it skips. Run: `docker compose run --rm dev pytest -m slow tests/test_live_supabase.py -q` → `1 skipped` without settings.

- [ ] **Step 8: Commit**

```bash
git add medrag/cloud/supabase.py medrag/services.py supabase/schema.sql tests/test_cloud_supabase.py tests/test_live_supabase.py
git commit -m "feat(cloud): Supabase backend, schema with RLS and RPCs, production service wiring"
```

---

### Task 13: Multi-user web app

**Files:**
- Create: `medrag/web/__init__.py` (empty), `medrag/web/components.py`, `medrag/web/local.py`, `medrag/web/multiuser.py`
- Modify: `app.py`, `tests/test_app.py`
- Test: `tests/test_web_multiuser.py`

**Interfaces:**
- Consumes: `Services`, `build_services`, `upload_book`, `delete_book` (Tasks 11–12); `AuthError`, `KeyRejected`, `UploadRejected`, `BackendUnavailable`.
- Produces: `medrag.web.components.render_message_body(msg: dict)`; `medrag.web.local.run()`; `medrag.web.multiuser.run()`, `get_services()`, `UNAVAILABLE_MESSAGE`. Session state: `user = {"id", "email"}`, widget keys `page`, `chat_book`, `login_email`, `login_password`, `login_submit`, `signup_email`, `signup_password`, `signup_submit`, `groq_key_input`, `save_key`, `remove_key`, `delete_<book_id>`, `logout`.

- [ ] **Step 1: Write the failing UI tests**

Create `tests/test_web_multiuser.py`:

```python
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from medrag.cloud.base import BackendUnavailable
from tests.fakes_cloud import make_services, ready_book

APP = str(Path(__file__).resolve().parent.parent / "app.py")
OWN = "gsk_" + "K8" * 12


@pytest.fixture
def env(monkeypatch):
    import streamlit as st

    import medrag.services

    services, llm = make_services()
    monkeypatch.setenv("MEDRAG_BACKEND", "memory")
    monkeypatch.setattr(medrag.services, "build_services", lambda settings, cloud: services)
    st.cache_resource.clear()
    return services, llm


def app_as(services, user=None, page=None):
    at = AppTest.from_file(APP, default_timeout=30)
    if user is not None:
        at.session_state["user"] = {"id": user.id, "email": user.email}
    if page is not None:
        at.session_state["page"] = page
    return at.run()


def text_of(at):
    return " ".join(e.value for e in [*at.markdown, *at.caption, *at.info, *at.error, *at.success, *at.warning])


def test_sign_up_logs_you_in(env):
    services, _ = env
    at = app_as(services)
    at.text_input(key="signup_email").input("new@example.org")
    at.text_input(key="signup_password").input("password123")
    at.button(key="signup_submit").click().run()
    assert not at.exception
    assert at.session_state["user"]["email"] == "new@example.org"
    assert "Signed in as new@example.org" in text_of(at)


def test_wrong_password_shows_error(env):
    services, _ = env
    services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services)
    at.text_input(key="login_email").input("a@example.org")
    at.text_input(key="login_password").input("wrong-password")
    at.button(key="login_submit").click().run()
    assert "Wrong email or password." in text_of(at)
    assert "user" not in at.session_state


def test_chat_answers_with_sources(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    ready_book(services, user.id)
    at = app_as(services, user)
    at.chat_input[0].set_value("What is tinnitus?").run()
    assert not at.exception
    assert "Tinnitus is ringing in the ear [1]." in text_of(at)
    assert any("p. 101" in e.label for e in at.expander)
    assert "Free questions today: 1/2." in text_of(at)


def test_chat_without_books_points_to_library(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services, user)
    assert "no ready books" in text_of(at)


def test_library_lists_and_deletes(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    book = ready_book(services, user.id)
    at = app_as(services, user, page="My library")
    assert "ENT Notes" in text_of(at) and "Ready · 4 pages" in text_of(at)
    at.button(key=f"delete_{book.id}").click().run()
    assert services.library.list(user.id) == []
    assert "No books yet." in text_of(at)


def test_settings_saves_masks_and_removes_key(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services, user, page="Settings")
    at.text_input(key="groq_key_input").input(OWN)
    at.button(key="save_key").click().run()
    assert f"gsk_…{OWN[-4:]}" in text_of(at) and OWN not in text_of(at)
    at.button(key="remove_key").click().run()
    assert services.vault.get(user.id) is None


def test_bad_key_is_explained(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services, user, page="Settings")
    at.text_input(key="groq_key_input").input("not-a-key")
    at.button(key="save_key").click().run()
    assert "should start with gsk_" in text_of(at)


def test_log_out(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services, user)
    at.button(key="logout").click().run()
    assert "user" not in at.session_state
    assert at.button(key="login_submit") is not None


def test_backend_outage_shows_banner(env, monkeypatch):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")

    def down(user_id):
        raise BackendUnavailable("paused")

    monkeypatch.setattr(services.library, "list", down)
    at = app_as(services, user)
    assert not at.exception
    assert "waking up or unavailable" in text_of(at)
```

In `tests/test_app.py`, add as the first line of the `app` fixture body:

```python
    monkeypatch.setenv("MEDRAG_BACKEND", "local")
```

- [ ] **Step 2: Run them to see them fail**

Run: `docker compose run --rm dev pytest tests/test_web_multiuser.py tests/test_app.py -q`
Expected: the multi-user tests FAIL (the app ignores `MEDRAG_BACKEND`); `test_app.py` still passes.

- [ ] **Step 3: Shared rendering — `medrag/web/components.py`**

```python
"""Rendering shared by the local and multi-user apps."""
import streamlit as st

from medrag.safety import EMERGENCY_MESSAGE


def render_sources(answer) -> None:
    cited = {c.chunk.id for c in answer.citations}
    for n, rc in enumerate(answer.chunks, start=1):
        c = rc.chunk
        status = "cited" if c.id in cited else "not cited"
        label = f"[{n}] {c.doc_title} — p. {c.page_label} · {c.section or 'section unknown'} ({status})"
        with st.expander(label):
            st.markdown(c.text)
            if st.session_state.get("debug"):
                st.caption(f"rerank={rc.rerank_score}  rrf={rc.rrf_score:.4f}  id={c.id}")


def render_message_body(msg: dict) -> None:
    """Everything inside one chat bubble: emergency banner, text, warnings and sources."""
    answer = msg.get("answer")
    if answer is not None and answer.emergency:
        st.error(EMERGENCY_MESSAGE)
    st.markdown(msg["content"])
    if answer is None:
        return
    if not answer.refused and not answer.citations:
        st.warning("This answer cites no sources. Verify it against the book before relying on it.")
    if st.session_state.get("debug"):
        if answer.standalone_question and answer.standalone_question != msg["question"]:
            st.caption(f"Searched for: {answer.standalone_question}")
        for warning in answer.warnings:
            st.caption(f"⚠ {warning}")
    if answer.chunks:
        render_sources(answer)
```

- [ ] **Step 4: Move the local app — `medrag/web/local.py`**

```python
"""Local single-user app over the Chroma index (no accounts). Used when no multi-user backend
is configured; this is the app the evaluation harness measures."""
import json
import time
from pathlib import Path

import streamlit as st

from medrag import pipeline as pipeline_module
from medrag.config import Settings
from medrag.safety import DISCLAIMER
from medrag.web.components import render_message_body

FEEDBACK_PATH = Path("data/feedback.jsonl")


@st.cache_resource(show_spinner="Loading models and index…")
def get_pipeline():
    return pipeline_module.build_pipeline(Settings.from_env())


def log_feedback(idx: int) -> None:
    msg = st.session_state.messages[idx]
    answer = msg["answer"]
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": msg["question"],
        "standalone_question": answer.standalone_question,
        "answer": answer.text,
        "refusal_reason": answer.refusal_reason,
        "cited_chunks": [c.chunk.id for c in answer.citations],
        "thumbs_up": st.session_state.get(f"fb_{idx}") == 1,
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def render_message(idx: int, msg: dict) -> None:
    with st.chat_message(msg["role"]):
        render_message_body(msg)
        answer = msg.get("answer")
        if answer is not None and answer.refusal_reason != "invalid_input":
            st.feedback("thumbs", key=f"fb_{idx}", on_change=log_feedback, args=(idx,))


def run() -> None:
    st.title("⚕️ MedRAG")
    st.caption("Answers grounded in the indexed medical textbook, with page citations you can check.")
    st.info(DISCLAIMER)

    try:
        rag = get_pipeline()
    except Exception as exc:
        st.error(f"MedRAG could not start: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        documents = rag.retriever.store.documents()
        selected = st.multiselect(
            "Search in", options=list(documents), default=list(documents), format_func=documents.get
        )
        st.toggle("Show retrieval details", key="debug")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    for i, message in enumerate(st.session_state.messages):
        render_message(i, message)

    if prompt := st.chat_input("Ask a question about the book…"):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Searching the book…"):
            try:
                answer = rag.answer(prompt, history=history, doc_ids=selected or None)
            except Exception as exc:
                st.session_state.messages.pop()
                st.error(f"Something went wrong while answering: {exc}")
                st.stop()
        st.session_state.messages.append(
            {"role": "assistant", "content": answer.text, "answer": answer, "question": prompt}
        )
        st.rerun()
```

- [ ] **Step 5: The multi-user app — `medrag/web/multiuser.py`**

```python
"""Multi-user MedRAG: accounts, a private library per user, and chat with one ready book."""
import streamlit as st

from medrag import services as services_module
from medrag.accounts import MIN_PASSWORD, AuthError
from medrag.cloud.base import BackendUnavailable
from medrag.config import CloudSettings, Settings
from medrag.keys import KeyRejected
from medrag.library import UploadRejected
from medrag.safety import DISCLAIMER
from medrag.web.components import render_message_body

UNAVAILABLE_MESSAGE = "The storage service is waking up or unavailable. Try again in a minute."
PAGES = ("Chat", "My library", "Settings")


@st.cache_resource(show_spinner="Loading models… the first start takes about a minute.")
def get_services():
    return services_module.build_services(Settings.from_env(), CloudSettings.from_env())


def run() -> None:
    st.title("⚕️ MedRAG")
    try:
        s = get_services()
    except Exception as exc:
        st.error(f"MedRAG could not start: {exc}")
        st.stop()
    try:
        user = st.session_state.get("user")
        if user is None:
            login_page(s)
            return
        with st.sidebar:
            st.caption(f"Signed in as {user['email']}")
            page = st.radio("Go to", PAGES, key="page")
            if st.button("Log out", key="logout"):
                st.session_state.clear()
                st.rerun()
        {"Chat": chat_page, "My library": library_page, "Settings": settings_page}[page](s, user)
    except BackendUnavailable:
        st.warning(UNAVAILABLE_MESSAGE)


def _enter(action) -> None:
    try:
        user = action()
    except AuthError as exc:
        st.error(str(exc))
        return
    st.session_state.user = {"id": user.id, "email": user.email}
    st.rerun()


def login_page(s) -> None:
    st.caption("Upload your own medical textbook and ask questions answered only from it, with page citations.")
    st.info(DISCLAIMER)
    log_in, sign_up = st.tabs(["Log in", "Sign up"])
    with log_in, st.form("login_form"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Log in", key="login_submit"):
            _enter(lambda: s.accounts.sign_in(email, password))
    with sign_up, st.form("signup_form"):
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        st.caption(f"At least {MIN_PASSWORD} characters. There is no password reset by email yet, so keep it safe.")
        if st.form_submit_button("Create account", key="signup_submit"):
            _enter(lambda: s.accounts.sign_up(new_email, new_password))


def _usage_line(s, user_id: str) -> str:
    cap = s.cloud.shared_daily_cap
    used = min(s.quota.used_today(user_id), cap)
    line = f"Free questions today: {used}/{cap}."
    if s.vault.has_key(user_id):
        line += " After that, your own Groq key is used."
    elif used >= cap:
        line += " Add your free Groq key in Settings to keep going."
    return line


def chat_page(s, user) -> None:
    uid = user["id"]
    ready = {b.id: b.title for b in s.library.list(uid) if b.status == "ready"}
    if not ready:
        st.info("You have no ready books yet. Upload one under **My library**.")
        return
    book_id = st.selectbox("Book", list(ready), format_func=ready.get, key="chat_book")
    st.info(DISCLAIMER)
    messages = st.session_state.setdefault("chats", {}).setdefault(book_id, [])
    for msg in messages:
        with st.chat_message(msg["role"]):
            render_message_body(msg)
    if messages and st.button("Clear chat"):
        messages.clear()
        st.rerun()
    st.caption(_usage_line(s, uid))
    if prompt := st.chat_input("Ask a question about this book…"):
        history = [{"role": m["role"], "content": m["content"]} for m in messages]
        messages.append({"role": "user", "content": prompt})
        with st.spinner("Searching the book…"):
            try:
                answer = s.chat.ask(uid, book_id, prompt, history)
            except BackendUnavailable:
                messages.pop()
                raise
            except Exception as exc:
                messages.pop()
                st.error(f"Something went wrong while answering: {exc}")
                return
        messages.append({"role": "assistant", "content": answer.text, "answer": answer, "question": prompt})
        st.rerun()


def library_page(s, user) -> None:
    uid = user["id"]
    c = s.cloud
    st.header("My library")
    st.caption(
        f"Upload a medical or health textbook as a PDF with selectable text; scanned PDFs can't be used. "
        f"Up to {c.max_books_per_user} books, each up to {c.max_pdf_mb} MB and {c.max_pages} pages."
    )
    with st.form("upload_form", clear_on_submit=True):
        file = st.file_uploader("PDF", type=["pdf"])
        title = st.text_input("Title (optional)")
        confirmed = st.checkbox("This is a medical or health book and I have the right to use it.")
        submitted = st.form_submit_button("Upload")
    if submitted:
        if file is None:
            st.error("Choose a PDF first.")
        else:
            try:
                with st.spinner("Checking the PDF…"):
                    book = services_module.upload_book(s, uid, title, file.name, file.getvalue(), confirmed)
                st.success(f"“{book.title}” was uploaded. Indexing has started and can take a few minutes.")
            except UploadRejected as exc:
                st.error(str(exc))
    books = s.library.list(uid)
    if any(b.status == "processing" for b in books):
        _live_books(s, uid)
    else:
        _render_books(s, uid, books)


@st.fragment(run_every=5)
def _live_books(s, uid: str) -> None:
    books = s.library.list(uid)
    _render_books(s, uid, books)
    if not any(b.status == "processing" for b in books):
        st.rerun()


def _render_books(s, uid: str, books) -> None:
    if not books:
        st.info("No books yet.")
        return
    for b in books:
        with st.container(border=True):
            st.markdown(f"**{b.title}**")
            if b.status == "ready":
                st.caption(f"Ready · {b.pages} pages · {b.chunks} passages · {b.filename}")
            elif b.status == "processing":
                st.caption(f"Indexing… {b.progress}")
            else:
                st.error(f"Indexing failed: {b.error}")
            if st.button("Delete", key=f"delete_{b.id}"):
                services_module.delete_book(s, uid, b.id)
                st.rerun()


def settings_page(s, user) -> None:
    uid = user["id"]
    cap = s.cloud.shared_daily_cap
    st.header("Settings")
    st.subheader("Your Groq API key")
    st.markdown(
        f"Everyone gets {cap} free questions a day on MedRAG's shared key. After that, MedRAG uses your own "
        "free Groq key: create one at [console.groq.com/keys](https://console.groq.com/keys). "
        "It is stored encrypted and never shown again in full."
    )
    with st.form("key_form", clear_on_submit=True):
        key = st.text_input("Groq API key", type="password", key="groq_key_input")
        if st.form_submit_button("Save key", key="save_key"):
            try:
                s.vault.save(uid, key)
                st.success("Key saved.")
            except KeyRejected as exc:
                st.error(str(exc))
    masked = s.vault.masked(uid)
    if masked:
        st.markdown(f"Saved key: `{masked}`")
        if st.button("Remove key", key="remove_key"):
            s.vault.remove(uid)
            st.rerun()
    else:
        st.caption("No key saved.")
    st.subheader("Today's usage")
    st.markdown(f"{min(s.quota.used_today(uid), cap)} of {cap} free questions used today (resets at 00:00 UTC).")
```

- [ ] **Step 6: Make `app.py` choose the app**

Replace the whole of `app.py` with:

```python
import streamlit as st

from medrag.config import CloudSettings

st.set_page_config(page_title="MedRAG", page_icon="⚕️", layout="wide")

if CloudSettings.from_env().backend == "local":
    from medrag.web import local as page
else:
    from medrag.web import multiuser as page

page.run()
```

- [ ] **Step 7: Run the tests**

Run: `docker compose run --rm dev pytest -q`
Expected: all pass. If AppTest cannot find a form submit button by `key`, look it up by label instead (`next(b for b in at.button if b.label == "Create account")`) in the test only; keep the keys in the app.

- [ ] **Step 8: Try it by hand in memory mode**

Run: `docker compose run --rm -p 8501:8501 -e MEDRAG_BACKEND=memory dev streamlit run app.py --server.address=0.0.0.0`
Open http://localhost:8501, sign up, upload a small text PDF (not the Dhingra book: anything you're allowed to use), watch it reach "Ready", ask a question, check the source card's page number, delete the book, then stop the container. Expected: no tracebacks; an off-topic question is refused.

- [ ] **Step 9: Commit**

```bash
git add app.py medrag/web tests/test_web_multiuser.py tests/test_app.py
git commit -m "feat(web): multi-user Streamlit app (log in, library, chat, settings); local app moved to medrag/web"
```

---

### Task 14: Production image

**Files:**
- Create: `Dockerfile`, `scripts/download_models.py`
- Modify: `.dockerignore`

**Interfaces:**
- Consumes: `Settings` defaults for model names; `app.py`.
- Produces: an image that serves the app on port 7860 with all three models baked in.

- [ ] **Step 1: Model download script**

Create `scripts/download_models.py`:

```python
"""Downloads the embedder, re-ranker and gate into the Hugging Face cache at image build time,
so a cold start doesn't download them."""
from medrag.config import Settings
from medrag.embeddings import SentenceTransformerEmbedder
from medrag.retrieval.rerank import CrossEncoderReranker

s = Settings()
SentenceTransformerEmbedder(s.embed_model)
CrossEncoderReranker(s.rerank_model)
CrossEncoderReranker(s.gate_model)
print("models cached:", s.embed_model, s.rerank_model, s.gate_model)
```

- [ ] **Step 2: Dockerfile**

Create `Dockerfile`:

```dockerfile
# Production image for a Hugging Face Docker Space (free CPU). Spaces run as uid 1000 and
# route traffic to app_port 7860 (set in README.md's front matter).
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/home/user/app
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER user
COPY --chown=user medrag medrag
COPY --chown=user scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py
COPY --chown=user app.py .
COPY --chown=user .streamlit .streamlit

EXPOSE 7860
# XSRF protection breaks file uploads inside the Hugging Face iframe.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
```

Append to `.dockerignore`:

```
tests
docs
legacy
review_gold.py
```

- [ ] **Step 3: Build and smoke-test it locally**

```bash
docker build -t medrag-prod .
docker run --rm -d --name medrag-prod -p 7860:7860 -e MEDRAG_BACKEND=memory medrag-prod
```

Wait for the health endpoint (about a minute while models load from the baked cache):

```bash
curl -s http://localhost:7860/_stcore/health
```

Expected: `ok`. Check that nothing was downloaded at startup: `docker logs medrag-prod 2>&1 | grep -ci "downloading"` → `0`. Then `docker stop medrag-prod`.

Check the image contains no secrets or books: `docker run --rm medrag-prod sh -c "ls -a; find / -name '*.pdf' -o -name '.env' 2>/dev/null | grep -v /usr/ | head"` → no `.env`, no PDFs.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile scripts/download_models.py .dockerignore
git commit -m "build: production Dockerfile for a Hugging Face Space with models baked in"
```

---

### Task 15: Deployment guide, README and spec update

**Files:**
- Create: `docs/deploy.md`
- Modify: `README.md`, `docs/superpowers/specs/2026-10-15-medrag-multiuser-design.md`

**Interfaces:**
- Consumes: everything above. Produces documentation only.

- [ ] **Step 1: Write `docs/deploy.md`**

```markdown
# Deploying MedRAG (free tiers)

MedRAG runs as a Hugging Face Docker Space (the app) plus a Supabase project (accounts, PDFs,
chunks, usage, encrypted keys). Both are free. You need a Groq API key for the shared quota.

## 1. Create the Supabase project
1. Sign in at supabase.com and create a new project (free plan). Pick a region near your users and save the database password somewhere safe.
2. In **Project Settings → API**, copy the **Project URL**, the **anon public** key and the **service_role** key. The service-role key bypasses all security rules: it goes only into Space secrets and your local `.env`, never into code, chat or screenshots.

## 2. Create the tables and the storage bucket
1. Open **SQL Editor → New query**, paste the contents of `supabase/schema.sql`, and run it. It is safe to run again.
2. Check **Table Editor** shows `books`, `chunks`, `usage` and `user_keys`, and **Storage** shows a private bucket named `books`.

## 3. Auth settings
1. **Authentication → Sign In / Providers → Email:** keep Email enabled. Turn **Confirm email** off (the app creates users already confirmed; this also stops Supabase sending mail it can't deliver on the free sender).
2. Password reset by email is not offered by the app. The free email sender is heavily rate-limited; this is a known limitation.

## 4. Generate the encryption secret
Run once and keep the output secret:

    docker compose run --rm dev python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

If you lose or change it, saved user Groq keys can no longer be read; users simply add them again.

## 5. Try it locally against Supabase (optional)
Add to `.env`: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `MEDRAG_FERNET_KEY` (and `GROQ_API_KEY`, already there). Then:

    docker compose run --rm dev pytest -m slow tests/test_live_supabase.py -q
    docker compose run --rm -p 8501:8501 dev streamlit run app.py --server.address=0.0.0.0

With Supabase variables in `.env`, the app starts in multi-user mode. Set `MEDRAG_BACKEND=local` to get the single-book app back.

## 6. Create the Space
1. At huggingface.co → **New Space**: SDK **Docker**, template **Blank**, hardware **CPU basic (free)**, visibility public or private.
2. **Settings → Variables and secrets → New secret** for each: `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `MEDRAG_FERNET_KEY`.
3. Create a Hugging Face **write** access token (Settings → Access Tokens).

## 7. Push the code
From the repo root:

    git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
    git push space v3-multiuser:main

Use your Hugging Face username and the write token as the password. The Space builds the `Dockerfile` (about 10 minutes the first time) and starts on port 7860 as declared in `README.md`'s front matter. Later deploys: push again.

## 8. Check it
Open the Space, sign up, upload a small text PDF you're allowed to use, wait for **Ready**, ask a question, open a source card, delete the book.

## Free-tier behaviour
- **The Space sleeps** after 48 hours without visitors; the next visit restarts it (about a minute while models load). Books being indexed at that moment are marked "interrupted"; users delete and re-upload them.
- **Supabase pauses** a free project after 7 days without activity. A paused project does not wake by itself: open the Supabase dashboard and click **Restore project** (free). Until then the app shows "The storage service is waking up or unavailable".
- **Limits** (change with environment variables, see `medrag/config.py`): 3 books per user, 50 MB / 1,000 pages per PDF, 60 books in total, 900 MB of PDFs, uploads pause above 400 MB of database, 10 shared-key answers per user per day.
- **Groq:** the shared key's free daily tokens are shared by all users; after their 10 daily answers, or if the shared key runs out, users add their own free key in Settings.
```

- [ ] **Step 2: README front matter and usage**

Add Hugging Face front matter at the very top of `README.md` (GitHub shows it as a small table; the Space needs it):

```markdown
---
title: MedRAG
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

and add a section after the existing introduction:

```markdown
## Multi-user mode

Anyone can sign up, upload their own medical textbook (a PDF with selectable text) and chat with it. Answers come only from that book, with page citations. See `docs/deploy.md` to run it on a free Hugging Face Space with Supabase.

- `MEDRAG_BACKEND=memory`: try multi-user mode locally with nothing saved.
- With `SUPABASE_URL` set: multi-user mode on Supabase.
- Neither: the original single-book app over the local Chroma index (used by the evaluation harness).
```

- [ ] **Step 3: Update the spec**

In `docs/superpowers/specs/2026-10-15-medrag-multiuser-design.md`:
- Change the status line to `**Status:** Approved 2026-10-15; implementation plan docs/superpowers/plans/2026-10-15-medrag-multiuser.md`.
- Append a section `## 9. Changes made during planning` containing the bullet list from this plan's "Deviations from the spec" section, verbatim.
- In §5, replace "`Dockerfile.prod`" with "`Dockerfile`", replace "the Space sleeps when idle" with "the Space sleeps after 48 hours without visitors", and replace "Supabase pauses after 7 days without activity and wakes on the next request" with "Supabase pauses after 7 days without activity and must be restored from the dashboard (free)".
- In §3, replace the "Ingestion crashes or the Space restarts mid-ingest" row's behaviour with: "At startup, every book still `processing` is marked `failed` ("interrupted") and its partial chunks are deleted; a crash inside a job marks that book failed and the worker continues."
- In §4, add `SUPABASE_ANON_KEY` to the secrets list.

- [ ] **Step 4: Final checks**

Run: `docker compose run --rm dev pytest -q` → all pass.
Run: `git status --short` and confirm no PDFs, `.env`, `data/chroma`, `data/eval_cache` or `data/eval_results` are staged.
Run: `git grep -nE "gsk_[A-Za-z0-9]{40,}|service_role\":|eyJhbGci" -- ':!tests'` → no matches (no real keys or JWTs committed).

- [ ] **Step 5: Commit**

```bash
git add docs/deploy.md README.md docs/superpowers/specs/2026-10-15-medrag-multiuser-design.md
git commit -m "docs: free-tier deployment guide, README multi-user section, spec updated with planning changes"
```

- [ ] **Step 6: Deploy (with the user)**

Deployment creates accounts and pushes code to an external service, so do it together with the user: follow `docs/deploy.md` steps 1–8. Then run the live test (`pytest -m slow tests/test_live_supabase.py`) against the real project and record the result in `docs/notes/deploy-check.md` (date, Space URL, test result; no keys).
