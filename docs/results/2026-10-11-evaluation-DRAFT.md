> **DRAFT** — this run includes gold items that have not been verified yet. Do not cite these numbers.

# MedRAG evaluation — 2026-10-11

Generator: `openai/gpt-oss-120b` · Judge: `qwen/qwen3.8-27b`

Gold items: in_scope 4, out_of_scope 4, near_miss 4, adversarial 4

## Answer quality

| system | in-scope n | answered | correct (of answered) | correct or partial | correct (overall) | book support | citation support | OOS refused | near-miss ok | adversarial ok | median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| medrag | 4 | 100.0% | 100.0% | 100.0% | 100.0% | 89.6% | 89.6% | 100.0% | 100.0% | 100.0% | 1.7 s |
| no_rag | 4 | 100.0% | 25.0% | 50.0% | 25.0% | 54.8% | — | 0.0% | 0.0% | 50.0% | 2.4 s |

*Book support*: share of answer claims stated by the retrieved textbook passages (the same passages for both systems). *Citation support*: share of cited claims supported by the passages they cite.

## Retrieval (page-level, in-scope items)

| variant | n | recall@1 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| dense | 90 | 75.6% | 94.4% | 95.6% | 0.837 | 0.863 |
| bm25 | 90 | 68.9% | 86.7% | 94.4% | 0.780 | 0.794 |
| v1_merge | 90 | 13.3% | 25.6% | 35.6% | 0.199 | 0.202 |
| rrf | 90 | 81.1% | 95.6% | 96.7% | 0.869 | 0.890 |
| rrf_rerank | 90 | 82.2% | 94.4% | 98.9% | 0.883 | 0.894 |
| rrf_rerank[medrag_t128] | 90 | 81.1% | 95.6% | 97.8% | 0.881 | 0.896 |
| rrf_rerank[medrag_t300] | 90 | 68.9% | 93.3% | 97.8% | 0.801 | 0.829 |

## Refusal threshold calibration

Chosen threshold (dev, in-scope pass ≥ 98.0%): **0.1206** (current setting 0.0100)

| split | in-scope passed | out-of-scope rejected |
|---|---|---|
| dev | 100.0% | 92.9% |
| test | 100.0% | 72.7% |

## Consistency

3 questions × 3 runs. Answer text cosine: **0.935** · cited-page Jaccard: **1.000**

