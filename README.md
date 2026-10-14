# MedRAG

Grounded question answering over medical textbooks. Answers come only from the indexed book, and every claim cites a numbered source with its printed page number and section.

## How it works

1. **Ingestion** (`python -m medrag.ingest`): PyMuPDF extracts each page. Running headers and footers are removed and printed page numbers inferred. Sections come from the PDF outline. Pages are split into ≤200-token sentence-aligned chunks, embedded with all-MiniLM-L6-v2 and stored in Chroma.
2. **Retrieval**: dense search and BM25 each return 30 candidates. These are fused with Reciprocal Rank Fusion and re-ranked by a cross-encoder; the top 5 are kept. If even the best passage scores below a threshold, MedRAG refuses without calling the LLM.
3. **Generation**: gpt-oss-120b (open-weight) on Groq answers from the numbered passages only and cites them as `[n]`. Citations to sources that don't exist are removed, and uncited sentences are flagged.
4. **Conversation**: follow-up questions are rewritten into standalone questions before retrieval.

## Setup

Development runs in Docker (Linux), which avoids Windows Smart App Control blocking native libraries. You need Docker Desktop.

```bash
copy .env.example .env          # then put your GROQ_API_KEY in .env
docker compose build
```

Put PDFs in `data/sources/` (they are gitignored) and index them. The index is stored on the `medrag-index` Docker volume.

```bash
docker compose run --rm dev python -m medrag.ingest data/sources/dhingra-ent-7e.pdf --title "Dhingra - Diseases of Ear, Nose and Throat (7th ed.)"
```

## Run

```bash
docker compose run --rm --service-ports dev streamlit run app.py --server.address 0.0.0.0
```

Then open http://localhost:8501.

## Test

```bash
docker compose run --rm dev python -m pytest              # fast unit tests
docker compose run --rm dev python -m pytest -m slow      # downloads models / calls Groq
docker compose run --rm dev python scripts/smoke.py       # real end-to-end check on known questions
```

## Project layout

See `docs/superpowers/specs/2026-10-11-medrag-v2-design.md`. Notes on the source PDF and the index are in `docs/notes/`. The v1 scripts behind the NGNDAI2026 submission are in `legacy/` and tagged `v1-paper`.

## Disclaimer

Educational tool. Not medical advice.

## Evaluation

The evaluation harness lives in `medrag/eval/`. Every Groq call is cached in `data/eval_cache/`, so runs can be resumed. On the free tier (8k tokens/min per model), a full run takes roughly 1.5–2 hours.

1. Draft the gold set (already done; it lives in `data/eval/gold.jsonl`):
   `docker compose run --rm dev python -m medrag.eval draft --n 90`
2. **Verify it.** Accept, edit or reject each item:
   `docker compose run --rm --service-ports dev streamlit run review_gold.py --server.address 0.0.0.0`
   (open http://localhost:8501)
3. Run the evaluation on verified items and write the report:
   ```bash
   docker compose run --rm dev python -m medrag.eval retrieval --collections medrag,medrag_t128,medrag_t300
   docker compose run --rm dev python -m medrag.eval calibrate
   docker compose run --rm dev python -m medrag.eval run
   docker compose run --rm dev python -m medrag.eval consistency
   docker compose run --rm dev python -m medrag.eval report
   ```

Reports are written to `docs/results/`. A report built from unverified items is marked DRAFT. The chunk-size variant collections (`medrag_t128`, `medrag_t300`) are built with `MEDRAG_COLLECTION` and `MEDRAG_CHUNK_MAX_TOKENS` / `MEDRAG_CHUNK_OVERLAP_TOKENS` passed to `python -m medrag.ingest`.
