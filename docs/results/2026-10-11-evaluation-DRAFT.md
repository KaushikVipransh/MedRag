> **DRAFT** — this run includes gold items that have not been verified yet. Do not cite these numbers.

# MedRAG evaluation — 2026-10-11

Generator: `openai/gpt-oss-120b` · Judge: `qwen/qwen3.8-27b`

Gold items: in_scope 4, out_of_scope 4, near_miss 4, adversarial 4

## Answer quality

| system | in-scope n | answered | graded | correct (of graded) | correct or partial | correct (overall) | book support | citation support | cited gold page | OOS refused | refusal precision | near-miss ok | adversarial ok | judge errors | LLM errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medrag | 4 | 100.0% | 4 | 100.0% | 100.0% | 100.0% | 89.6% | 89.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 0 |
| no_rag | 4 | 100.0% | 4 | 25.0% | 50.0% | 25.0% | 54.8% | — | 0.0% | 0.0% | — | 0.0% | 50.0% | 0 | 0 |

*Correct (overall)*: correct answers over in-scope items that were refused or graded (items whose judge failed are excluded and counted under judge errors). *Book support*: mean, over answered items, of the share of an answer's claims stated by the retrieved textbook passages (the same passages for both systems). *Citation support*: share of cited claims supported by the passages they cite. *Cited gold page*: answered items citing at least one gold page. *Refusal precision*: share of refusals that were for questions the book does not answer. *OOS refused* counts only explicit refusals.

Paired book support on the 4 in-scope items both systems answered: medrag **89.6%** vs no_rag **54.8%**.

## Retrieval (in-scope items; metrics over the top-k unique pages)

| variant | n | recall@1 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| dense | 90 | 75.6% | 95.6% | 95.6% | 0.837 | 0.867 |
| bm25 | 90 | 68.9% | 86.7% | 94.4% | 0.780 | 0.794 |
| v1_merge | 90 | 13.3% | 26.7% | 35.6% | 0.199 | 0.206 |
| rrf | 90 | 81.1% | 95.6% | 96.7% | 0.869 | 0.890 |
| rrf_rerank | 90 | 82.2% | 95.6% | 98.9% | 0.883 | 0.898 |
| rrf_rerank[medrag_t128] | 90 | 81.1% | 96.7% | 97.8% | 0.881 | 0.901 |
| rrf_rerank[medrag_t300] | 90 | 68.9% | 95.6% | 97.8% | 0.801 | 0.838 |

## Refusal threshold calibration

Chosen threshold (dev, in-scope pass ≥ 98.0%): **0.1206** (current setting 0.0100)

| split | in-scope passed | out-of-scope rejected |
|---|---|---|
| dev | 100.0% | 92.9% |
| test | 100.0% | 72.7% |

## Consistency

3 questions × 3 runs. Answer text cosine: **0.935** · cited-page Jaccard: **1.000**

