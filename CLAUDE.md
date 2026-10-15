# MedRAG — context for Claude

## What this is
A medical chatbot that answers only from a textbook, with page citations, and refuses rather than hallucinating. It is being turned into a free multi-user app: people sign up, upload their own medical textbook (a text PDF), and chat with it. The research paper (v1, NGNDAI2026) is secondary. The Dhingra ENT book is test data only.

## Current state (2026-10-15)
- Branch `v3-multiuser` (from `v2-accuracy`). Neither branch is pushed yet.
- Spec (approved): `docs/superpowers/specs/2026-10-15-medrag-multiuser-design.md`
- Plan: `docs/superpowers/plans/2026-10-15-medrag-multiuser.md` (Tasks 0–15)
- **Next step:** execute the plan **natively** (the owner chose this) with `superpowers:executing-plans`, starting at Task 0. When all tasks are done, one fresh reviewer checks the whole branch.
- Task 0 runs the 40-question dev-subset check, which needs Groq quota. If it aborts with `QuotaExhausted`, note that and continue; the run resumes later.
- Task 15 Step 6 (creating the Supabase project and Hugging Face Space, pushing) is done **with the owner**, not alone.
- Test baseline before the plan: 201 passed.

## How to run things
- The host is Windows ARM64 with Smart App Control ON, which blocks native scipy/grpc DLLs. **Run everything in Docker:** `docker compose run --rm dev <command>`. Never suggest a native Windows venv.
- Tests: `docker compose run --rm dev pytest -q` (`slow` tests are excluded by default; run them with `-m slow`).
- In Git Bash, prefix container paths with `MSYS_NO_PATHCONV=1`.
- The Chroma index lives on Docker volume `medrag-index` (`MEDRAG_CHROMA_DIR=/var/medrag/chroma`); the model cache on volume `hf-cache`.
- Evaluation: `python -m medrag.eval ...` (subset: `--ids-file data/eval/dev_subset.txt`).

## Hard rules
- **Free tiers only.** The owner will not pay for anything: Hugging Face free CPU Space, Supabase free tier, Groq free tier.
- **The GitHub repo is PUBLIC** (github.com/KaushikVipransh/MedRag). Never commit PDFs, `.env`, the index, `data/eval/evidence.jsonl`, `data/eval_cache/` or `data/eval_results/`. Never print API keys.
- The test book is copyrighted (it carries a mebooksfree watermark). Keep the book and its index private.
- Commit or push only when asked; branch before working on `main`.

## Key technical facts
- **Models:**
  - generator `openai/gpt-oss-120b` on Groq (Llama 3.3 is retired);
  - judge and drafter `qwen/qwen3.8-27b`;
  - embedder `all-MiniLM-L6-v2`;
  - re-ranker `BAAI/bge-reranker-base`;
  - relevance gate `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Groq free tier, per model:** about 8k tokens/min, 1,000 requests/day, 200k tokens/day. Reasoning models need `max_tokens` around 2048.
- **Retrieval:** dense + BM25 → RRF → bge re-rank; the gate's best score over the candidate pool decides refusal. The calibrated threshold is about 0.045 (weakest verified in-scope question 0.057, strongest out-of-scope 0.037).
- **Ingestion:** printed page labels, margin stripping, TOC sections and topic headings; chunks are 200 tokens with 40 overlap; `context_text` prefixes "Chapter — Topic:".
- **Baseline on 129 verified questions:** 90.5% correct of those graded; out-of-scope, near-miss and adversarial handling 100%.
