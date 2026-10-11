# Smoke run — 2026-10-11

`docker compose run --rm dev python scripts/smoke.py` against the real index (2,400 chunks).

**The Groq key in `.env` is rejected (401 Invalid API Key)**, so LLM generation could not run. Everything up to the LLM call ran for real: input validation, retrieval, RRF, cross-encoder re-ranking and the retrieval-level refusal.

| scope | question | top rerank | outcome |
|---|---|---|---|
| in | What is Ménière's disease? | 0.9998 | passed threshold → LLM (401) |
| in | What is tinnitus retraining therapy? | 0.9816 | passed threshold → LLM (401) |
| in | What are the orbital complications of sinusitis? | 0.9996 | passed threshold → LLM (401) |
| in | What is FESS? | 0.9992 | passed threshold → LLM (401) |
| in | What are the parts of the nasal septum? | 0.9999 | passed threshold → LLM (401) |
| out | COVID-19 vaccine schedule 2025 | 0.0000 | **refused: low_relevance** ✔ (no LLM call) |
| out | How do ACE inhibitors lower blood pressure? | 0.0020 | **refused: low_relevance** ✔ (no LLM call) |
| out | Symptoms of appendicitis? | 0.9642 | passed threshold → LLM (401) |

Latency per question: 0.4–0.7 s for retrieval and re-ranking on CPU (arm64, Docker).

## Verdicts
1. In-scope questions answered with correct citations: **pending a valid Groq key.**
2. FESS answer must not describe a flexible endoscope: **pending a valid Groq key.**
3. Out-of-scope refused: **2/3 verified** at the retrieval level. Appendicitis is a *near-miss*: the book really does mention it (printed p. 287, pharyngoconjunctival fever "pain in abdomen, mimicking appendicitis"; p. 293, tonsillitis "abdominal pain … simulating … acute appendicitis"). The re-ranker is right to score it high, so the correct outcome depends on the LLM giving a partial answer or refusing. That is exactly the near-miss category Plan 2 must measure. v1's "100% refusal" on this question relied on the LLM alone.
4. Threshold sanity: every in-scope top score (≥ 0.98) is far above `0.01`, and clear out-of-scope questions score ≤ 0.002. The default threshold is reasonable; Plan 2 calibrates it.

## To finish once the key is replaced
```
docker compose run --rm dev python -m pytest -m slow
docker compose run --rm dev python scripts/smoke.py
```
Then do the manual UI checklist from plan Task 15 Step 7 (follow-up rewriting, emergency banner, feedback logging, clear chat).
