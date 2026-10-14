# MedRAG v2 — Design Spec

**Date:** 2026-10-11
**Owner:** Vipransh Kaushik (solo)
**Status:** Approved scope, pre-implementation

## 1. Goal

Rebuild MedRAG into a system that is *correct and verifiable*, not just functional: answers come only from the indexed book, every claim carries a citation a reader can check against the printed page, and an automated evaluation measures whether that is actually true.

Deployment (Hugging Face Spaces) is **out of scope** for this spec; the design only has to keep it easy (single Python process, config via env vars, no committed copyrighted data).

## 2. Decisions

| Topic | Decision |
|---|---|
| Corpus | Single book first (Dhingra ENT, 7th ed.). Every layer is multi-document aware (doc_id in metadata, per-document filtering, doc title in citations). |
| Evaluation labels | Gold dataset is drafted with tool assistance and **verified by the owner alone**. Each item carries a verbatim evidence quote + page so verification is fast. |
| LLM provider | Groq. Generator model configurable (default `llama-3.3-70b-versatile`). Judge model for evaluation is a *different* Groq-hosted model family, chosen when Plan 2 starts by listing available models. |
| UI | Streamlit, calling the pipeline as a library. FastAPI can be added at deployment time without touching the pipeline. |
| Retrieval | Dense (MiniLM) + BM25 → Reciprocal Rank Fusion → cross-encoder re-rank → top 5. |
| Python | 3.12 preferred (3.13 acceptable if all deps install). CPU only. |

## 3. Problems in v1 this design fixes

1. Hybrid merge sorts cosine similarity and max-normalised BM25 together (incomparable scales). → RRF.
2. BM25 tokenisation is `str.split()` (case/punctuation sensitive). → normalised tokenizer.
3. Citations use PDF page index, not printed page number. → page-label inference.
4. Running headers/footers and page numbers pollute chunks; skip filter drops any page with "index"/"references" in the first 80 chars. → margin-line removal + structural non-content detection.
5. Chunking ignores document structure. → section metadata from PDF outline (font-size fallback); chunks never span pages, so page citations are exact.
6. No multi-turn retrieval despite "multi-turn chat" claim. → follow-up condensation.
7. Grounding enforced only by prompt; ungrounded content (e.g. "flexible tube" for FESS) passed evaluation. → numbered-source citations, citation validation, uncited-sentence warnings, retrieval-level refusal, faithfulness metric (Plan 2).
8. Evaluation is manual copy-paste, relevancy ground truths partly copied from the bot's own answers, refusals hand-labelled, false refusals unmeasured. → automated harness over a verified gold set (Plan 2).
9. No error handling, no input validation, no disclaimer. → safety module + graceful failures.

## 4. Architecture

```
            ┌──────────── ingestion (offline) ────────────┐
PDF(s) ──▶ read_pdf ─▶ page labels ─▶ strip headers/footers ─▶ sections ─▶ chunk (≤200 tokens, within page)
                                                                              │
                                                              embed (MiniLM) ─▶ Chroma (doc_id, page_label, section)

            ┌──────────── query (online) ─────────────────┐
question ─▶ validate ─▶ condense follow-up ─▶ dense top30 ┐
                                              BM25 top30 ┴─▶ RRF ─▶ cross-encoder ─▶ top5
                                                                     │
                       best score < threshold? ── yes ─▶ refuse (low_relevance)
                                                                     │ no
                                    Groq LLM with numbered sources [1..5] ─▶ NOT_FOUND? ─▶ refuse (not_in_context)
                                                                     │
                                    validate citations, flag uncited sentences ─▶ Answer
```

### Package layout

```
medrag/
  config.py            Settings (env-driven)
  types.py             Chunk, RetrievedChunk, Citation, Answer
  embeddings.py        Embedder protocol + SentenceTransformerEmbedder
  store.py             ChunkStore (Chroma wrapper, doc-aware)
  ingest/
    pdf.py             read_pdf, read_font_lines
    clean.py           margin-line removal, line joining, non-content detection
    labels.py          printed page-label inference
    sections.py        sections from TOC / font sizes
    chunking.py        token-bounded sentence packer with overlap
    build.py           build_chunks, ingest_pdf
    __main__.py        CLI: python -m medrag.ingest <file-or-dir>
  retrieval/
    bm25.py            tokenize, BM25Index
    fusion.py          reciprocal_rank_fusion
    rerank.py          Reranker protocol + CrossEncoderReranker
    retriever.py       HybridRetriever
  generation/
    llm.py             LLMClient protocol, GroqClient, LLMError
    prompts.py         system prompt, source formatting, refusal constants
    citations.py       parse/validate [n] markers, uncited sentences
    generator.py       Generator
  conversation.py      condense_question
  safety.py            validate_question, is_emergency, disclaimer text
  pipeline.py          MedRAG facade + build_pipeline
app.py                 Streamlit UI
scripts/               inspect_pdf.py, smoke.py
legacy/                v1 scripts (also tagged v1-paper in git)
data/sources/          PDFs (gitignored)
data/chroma/           index (gitignored)
```

### Key contracts

- `MedRAG.answer(question, history=None, doc_ids=None) -> Answer` is the **only** entry point; the UI and the evaluation harness both call it.
- `Answer.refusal_reason ∈ {None, "invalid_input", "low_relevance", "not_in_context", "llm_error"}`.
- Answer text cites sources as `[n]`; `Answer.citations` maps each valid `n` to its `Chunk` (doc title, printed page, section, text).

## 5. Safety behaviour

- Questions are trimmed, control characters removed, whitespace collapsed; empty or > 1000 chars is rejected with a message.
- First-person emergency phrases (can't breathe, choking, swallowed battery, severe bleeding, suicidal) set `Answer.emergency`, and the UI shows a "seek emergency care" banner above the answer.
- A permanent disclaimer: educational tool, not medical advice.
- Retrieved text is labelled as reference material in the prompt; the model is told to ignore instructions inside sources or the question that conflict with its rules.

## 6. Evaluation (Plan 2 — written after ingestion lands, because the gold set is drafted from real chunks)

- **Gold set (~120 items):** ~80 in-scope questions spread across ear / nose / throat / head-and-neck chapters, each with answer key, gold printed page(s) and verbatim evidence quote; ~25 out-of-scope (general medicine not in the book); ~15 near-miss and adversarial (ENT-adjacent but absent, prompt injection, false-premise questions).
- **Retrieval metrics:** Recall@5, MRR, nDCG@5 against gold pages.
- **Generation metrics:** claim-level faithfulness (judge LLM checks each claim against cited passages), correctness vs answer key, citation accuracy (cited page ∈ gold pages), refusal precision/recall (false refusals counted), consistency (secondary).
- **Ablations:** no-RAG LLM baseline; dense-only; BM25-only; v1 merge; RRF; RRF + re-rank; chunking variants.
- One command runs everything, with cached LLM calls and throttling for Groq rate limits; results saved as JSON/CSV.
- The rerank refusal threshold is calibrated on the gold set here.

## 7. Constraints

- PDFs (book and paperwork) and `.env` are never committed. The book is copyrighted; the paperwork contains personal phone numbers.
- No GPU assumed; all models must run on CPU.
- Old evaluation numbers remain valid for v1 (tag `v1-paper`); v2 produces its own results table.

## 8. Evaluation design decisions (added 2026-10-11, before Plan 2)

- **Models.** Generator: `openai/gpt-oss-120b`; Groq retired `llama-3.3-70b-versatile`. Drafter and judge: `qwen/qwen3.8-27b`, a different model family from the generator, so the system under test never writes or grades its own questions. Both models are configurable.
- **Groq free-tier limits** (measured): 8,000 tokens/min and 1,000 requests/day per model. Every LLM call in evaluation goes through a disk cache and a rate-limit waiter. Runs can be resumed, and a re-run costs nothing for calls already made.
- **Relevance is page-level.** Gold labels are printed page numbers, not chunk ids, so retrieval metrics compare fairly across chunking variants.
- **Gold set composition:**
  - ~90 drafted in-scope items, expected to shrink to ~80 after review.
  - 25 out-of-scope items (general medicine).
  - 10 near-miss items (ENT-adjacent, mentioned in passing, or post-publication).
  - 6 adversarial items (prompt injection, false premise).
- **Workflow order (owner's decision).** Build all tooling first. The owner verifies the gold set afterwards in a review UI. Until then, runs require `--include-unverified`, and their reports are stamped DRAFT.
- **What is committed:** `data/eval/gold.jsonl` (questions, answer keys, gold pages) and the reports in `docs/results/`. **What is not committed:** `data/eval/evidence.jsonl` (verbatim book quotes), the LLM cache, and per-item results.
- **Book support.** For both the RAG system and the no-RAG baseline, the judge checks each answer claim against the same retrieved book passages. This makes "share of claims the book supports" comparable between the two.
