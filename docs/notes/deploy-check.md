# Deploy check — 2026-10-16

**App:** https://medragbot.streamlit.app/ (Streamlit Community Cloud, branch `main`, Python 3.14)
**Store:** Supabase project `medrag`, region us-east-1 (free plan)

Hugging Face Docker Spaces were not used: since mid-2026 free accounts get a 402 ("requires a PRO subscription") for Docker/Gradio Spaces on CPU.

## Checks

- `pytest -m slow tests/test_live_supabase.py` against the project (MEDRAG_BACKEND=supabase): **1 passed** — sign-up, sign-in, upload, indexing, answer, usage RPC, cross-user isolation, delete.
- Pagination: 2,345 chunk rows inserted and loaded back in order; embeddings round-trip as float lists; delete cascades.
- Full suite under Python 3.14.8 with the requirements installed by uv: **335 passed**.
- Headless browser (Playwright) against the live URL: sign up → upload a 4-page generated PDF → "Ready · 4 pages · 4 passages" → "What is tinnitus?" answered with citation [1] to p. 101 → off-topic question refused ("not found") → usage "1/10" → delete → "No books yet." **E2E OK.**
- Test accounts removed afterwards; production left with 0 users, 0 books, 0 chunks, 0 stored files.

## Notes

- Clicking **Upload** before the browser has finished sending the file shows "Choose a PDF first."; waiting for the file chip avoids it.
- A gate refusal still lists the retrieved passages as "not cited" source cards (same as the local app).
