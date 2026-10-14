# MedRAG v3 — Multi-user "bring your own medical book" — Design Spec

**Date:** 2026-10-15
**Owner:** Vipransh Kaushik
**Status:** Sections 1–2 approved in conversation; sections 3–7 drafted for review
**Builds on:** `2026-10-11-medrag-v2-design.md` (single-book v2, branch `v2-accuracy`)

## 0. Intent

People sign up, upload their own **medical** textbook (a PDF with a text layer), and chat with a bot that answers **only from that book**, with page citations, and refuses rather than hallucinating. The Dhingra ENT book becomes test data only. Everything must run on **free tiers**: no paid hosting, no paid APIs.

| Decision | Choice |
|---|---|
| Who uploads / who sees | **Private per-user libraries** with accounts; a user only ever sees their own books |
| LLM quota | **Shared app key first** (daily cap per user), then **the user's own free Groq key** as fallback |
| Hosting | **Hugging Face Space** (free CPU: 2 vCPU, 16 GB RAM) for the app, plus **Supabase** free tier (Auth, Postgres, Storage) for durable data |
| Book domain | **Medical and health books only**; medical safety features stay always on |
| Scanned PDFs | **Rejected** with a clear message (no OCR) |
| Architecture | **Approach A:** Streamlit app; Supabase as the durable store; each selected book is searched **in memory** with today's retrieval stack |

## 1. Architecture and components

```
Browser ── Streamlit app (HF Space) ── Groq (shared key → user's own key)
                 │
                 └── Supabase: Auth · Storage bucket "books" · Postgres tables
```

New units, each with one job and a narrow interface:

| Unit | Responsibility |
|---|---|
| `medrag/accounts` | Sign up, log in and log out via Supabase Auth (email and password); exposes the current `user_id` and session |
| `medrag/library` | A user's books: upload, list, delete, limit checks. **Every query takes `user_id` and filters by it** |
| `medrag/ingest/job` | Runs ingestion in a background worker with progress; sets book status `processing` / `ready` / `failed` (+ reason) |
| `medrag/store_pg` | Postgres-backed `ChunkStore` with today's interface (`add`, `all_chunks`, `delete_document`, `query`, `count`, `documents`). Loads one book's chunks into memory, with a small cache of recently used books, and searches there |
| `medrag/quota` | Per-user daily usage of the shared key; picks the Groq key for each question |
| `medrag/keys` | Encrypts and decrypts users' Groq keys (Fernet; the secret lives only in Space settings); never shows a saved key again |

**Unchanged and reused:** PDF ingestion (cleaning, printed page labels, sections, topic headings, chunking), BM25, the bge re-ranker, the MiniLM relevance gate, citation handling, safety checks, `MedRAG.answer()`, and the evaluation harness, which keeps running locally against the Chroma store.

**Pages:**
- **Login / sign-up**
- **My library:** upload, status, delete
- **Chat:** pick one of your ready books, then chat, with source cards as today
- **Settings:** add or remove your Groq key, and see today's usage

## 2. Data model and data flow

**Tables.** Every row carries `user_id`; Row-Level Security (RLS) policies restrict each user to their own rows.

| Table | Columns |
|---|---|
| `books` | `id uuid pk, user_id, title, filename, storage_path, pages, chunks, status ∈ {processing, ready, failed}, progress text, error text, created_at` |
| `chunks` | `id text pk, book_id fk → books (on delete cascade), user_id, text, page_index, page_label, section, topic, embedding float4[384]` |
| `usage` | `user_id, day date, shared_key_questions int` (primary key `user_id, day`) |
| `user_keys` | `user_id pk, encrypted_groq_key text, created_at` |

**Storage bucket `books`** (private) holds PDFs at `{user_id}/{book_id}.pdf`.

**Upload flow:**
1. **Pre-checks:**
   - the user has fewer than 3 books;
   - the file is at most 50 MB and 1,000 pages;
   - the PDF opens (not encrypted or corrupt);
   - it has a text layer: if ≥ 20% of pages yield no usable text, reject it as scanned;
   - the global book cap (§6) has not been reached.
2. **Store:** upload the PDF to the bucket and insert a `books` row with `status=processing`.
3. **Ingest in the background:** extract, clean, label pages, add sections and topics, chunk, embed, and insert the chunks in batches. `progress` on the book row is updated as it goes.
4. **Finish:**
   - On success, set `status=ready` with the page and chunk counts.
   - On any error, set `status=failed` with a human-readable reason and delete any partial chunks. The user can delete the book and try again.

**Question flow:**
1. The user picks a `ready` book from their own library.
2. The app loads that book's chunks into memory once and caches them (LRU, up to 4 books). Dense search is brute-force cosine in numpy; BM25 is built in memory as today.
3. `MedRAG.answer()` runs unchanged, scoped to that book.
4. Questions refused before the LLM call cost nothing and do not count toward the cap. These are refusals by the relevance gate (`low_relevance`) and rejected input (`invalid_input`).
5. Before the LLM call, `quota` picks the key:
   - **the shared key** while today's count is below 10;
   - **otherwise the user's own key**, if one is saved;
   - **otherwise** the user sees: *"You've used today's 10 free questions. Add your free Groq key in Settings to keep going."*

   The count goes up only when an answer is actually generated on the shared key.

**Deleting a book** removes the PDF, the chunks (by cascade) and the row, and evicts it from the in-memory cache.

## 3. Error handling

| Situation | Behaviour |
|---|---|
| Scanned or image-only PDF | Rejected before upload: *"This PDF has no text layer (it looks scanned). Upload a text-based PDF."* |
| Encrypted, corrupt or non-PDF file | Rejected: *"This file couldn't be opened as a PDF."* |
| Over a limit (books, size, pages, global cap) | Rejected, with the limit named and the current usage shown |
| Ingestion crashes or the Space restarts mid-ingest | Rows still `processing` with no progress update for more than 15 minutes are marked `failed` ("interrupted"); partial chunks are deleted |
| Shared key hits Groq's own daily limit (`QuotaExhausted` / 429 with a long wait) | Fall back to the user's key if they have one; otherwise *"The free shared quota is used up for today. Add your free Groq key in Settings."* |
| User's own key is invalid or exhausted | *"Your Groq key was rejected / has hit its daily limit"*, with a link to Settings. No silent fallback to the shared key beyond the user's cap |
| Supabase unreachable or paused | A clear banner (*"Storage service is waking up / unavailable; try again in a minute"*); no traceback |
| Other users' data | Never reachable: `library` and `store_pg` require `user_id` on every call, and RLS is defence in depth |

## 4. Security and privacy

- **Secrets:** `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `GROQ_API_KEY` (shared) and `MEDRAG_FERNET_KEY` live only in Hugging Face Space secrets and a local `.env`, and are never committed.
- **Service key handling:** the service-role key is used server-side only (Streamlit runs on the server and nothing is sent to the browser). Because it bypasses RLS, every data-access function takes an explicit `user_id`, and tests assert cross-user isolation. RLS policies still exist for any future anon-key access.
- **Users' Groq keys:**
  - encrypted at rest with Fernet;
  - shown masked after saving (`gsk_…abcd`);
  - removable at any time;
  - never logged.
- **Uploaded books:**
  - private bucket;
  - only the owner's `user_id` can list, chat with or delete them;
  - the upload screen asks users to confirm they have the right to use the book.
- **Email verification:** Supabase's free built-in email sender is heavily rate-limited, so email confirmation is **disabled**. Sign-up is email plus password. Password reset by email is limited by the same sender and is documented as a known limitation.
- **Medical safety:** the disclaimer and emergency banner stay always on, and the upload screen states the app is meant for medical and health books.

## 5. Deployment

- **`Dockerfile.prod`** (Hugging Face Docker Space):
  - Python 3.12 slim with CPU-only torch;
  - the three models pre-downloaded at build time (MiniLM embedder, bge re-ranker, MiniLM gate) so a cold start doesn't download them;
  - runs `streamlit run app.py` on port 7860.
- **`supabase/schema.sql`:** tables, indexes, RLS policies and the storage bucket, applied once in the Supabase SQL editor.
- **`docs/deploy.md`:** a step-by-step guide covering:
  1. create the Supabase project;
  2. run the schema;
  3. disable email confirmation;
  4. create the Hugging Face Space;
  5. add the secrets;
  6. push.
- **Free-tier behaviour:**
  - the Space sleeps when idle; the first request after sleep takes about 1 minute to load the models;
  - Supabase pauses after 7 days without activity and wakes on the next request.
- **Local development** keeps the existing Docker setup. With no Supabase variables set, the app runs in **local mode**: a single built-in user, the Chroma store and the shared key only. All current tests and the evaluation harness keep working.

## 6. Capacity and limits (free tiers)

| Resource | Free limit | Design response |
|---|---|---|
| Supabase database | 500 MB | ≈ 6 MB per 500-page book → **global cap of 60 books** (configurable); uploads are disabled with a message when it's reached |
| Supabase storage | 1 GB | 50 MB per PDF × 60 books worst case 3 GB → also enforce **total stored PDF bytes ≤ 900 MB** |
| Groq shared key | ~200k tokens/day per model | 10 generated answers per user per day; users' own keys beyond that |
| Space CPU | 2 vCPU | **One ingestion at a time** (a queue; others show "queued"); bge re-ranking takes a few seconds per question |

## 7. Testing

- **Unit tests** with an in-memory fake Supabase client (no network):
  - `library` limits and user isolation (user B can never list, load or delete user A's book);
  - `store_pg` parity with `ChunkStore` behaviour;
  - `quota` key selection and counting rules (refusals don't count);
  - `keys` encrypt/decrypt round-trip and masking;
  - ingestion job state transitions (success, failure, interrupted, scanned-PDF rejection).
- **UI tests** with Streamlit AppTest for login, library, chat and settings, against fakes.
- **Live test** (`@pytest.mark.slow`, skipped without `SUPABASE_URL`): sign up a throwaway user, upload a small generated PDF, ask a question, delete it.
- **Accuracy is unchanged and still measured** by the existing evaluation harness on the Dhingra test book, including the 40-item dev subset within free quota.

## 8. Scope

**In scope:** everything above.

**Out of scope (YAGNI for now):**
- OCR;
- sharing books between users;
- payments;
- admin dashboard;
- non-medical books;
- password reset by email beyond Supabase's limited sender;
- a separate API or frontend (approach C);
- pgvector (approach B).

**Prerequisites carried over from v2-accuracy:**
- lower the relevance threshold to about 0.045 (the weakest verified in-scope question scores 0.057, the strongest out-of-scope one 0.037);
- run the 40-item dev-subset check once the Groq quota resets.

Both are done before the multi-user work starts.
