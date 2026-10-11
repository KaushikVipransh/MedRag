# Smoke run — 2026-10-11 (live, after key rotation)

`docker compose run --rm dev python scripts/smoke.py` against the real index (2,402 chunks), generator `openai/gpt-oss-120b` on Groq.

**Model change:** Groq has retired `llama-3.3-70b-versatile`; the API returns 404 `model_not_found`. The currently available chat models are `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` and `allam-2-7b`. The default is now `openai/gpt-oss-120b`: open-weight under Apache-2.0, so the "open-source LLM" framing still holds. Qwen is reserved as the cross-family judge for Plan 2.

| scope | question | top rerank | outcome | cited printed pages |
|---|---|---|---|---|
| in | What is Ménière's disease? | 0.9998 | answered | Ménière's chapter (pp. 111–115) |
| in | What is tinnitus retraining therapy? | 0.9816 | answered | 146 |
| in | Orbital complications of sinusitis? | 0.9996 | answered: pre-septal oedema, subperiosteal abscess, orbital cellulitis, orbital abscess, superior orbital fissure syndrome, orbital apex syndrome | 224–225 |
| in | What is FESS? | 0.9992 | answered: treatment of choice for chronic rhinosinusitis; 0°/30°/70° endoscopes; **no "flexible"** ✔ | sinus surgery chapter |
| in | Parts of the nasal septum? | 0.9999 | answered: columellar, membranous, septum proper | nasal anatomy chapter |
| out | COVID-19 vaccine schedule 2025 | 0.0000 | refused: `low_relevance` (no LLM call) ✔ | — |
| out | ACE inhibitors | 0.0020 | refused: `low_relevance` (no LLM call) ✔ | — |
| out | Symptoms of appendicitis | 0.9642 | refused: `not_in_context` ✔ (the LLM recognised that the book mentions appendicitis only as a mimic) | — |

Latency: 1.4–2.4 s per answered question (warm); retrieval-level refusals take about 0.5 s.

## Integration bug found and fixed
gpt-oss cites as `【1】` or `【1†L1-L3】` rather than `[1]`, so the first run flagged 3 of 5 answers as "cites no sources". `normalize_markers` now converts these, and the prompt asks for plain square brackets. Both are covered by tests (`test_alternative_citation_styles_are_normalised`, `test_lenticular_citations_from_reasoning_models_count_as_citations`). After the fix, every answer carries valid citations. The remaining warnings are genuine: an uncited concluding sentence, e.g. *"These elements together help the patient 'ignore' the tinnitus…"*.

## Conversational checks (live)
- Follow-up "how is it treated?" after "What is tinnitus?" was rewritten to **"How is tinnitus treated?"** and answered with citations to pp. 145, 146 and 56 ✔
- "my child swallowed a button battery" → `emergency=True`, so the UI shows the banner ✔
- "what is it?" (no history) → `invalid_input`: "Please ask a specific question…" with no LLM call ✔

## Verdicts
1. In-scope questions answered with correct citations ✔
2. FESS no longer describes a flexible endoscope ✔
3. Out-of-scope refused 3/3 ✔
4. Threshold: in-scope top scores are ≥ 0.98 and clear out-of-scope scores ≤ 0.002, so 0.01 is a sound default. Plan 2 will calibrate it on the gold set.
