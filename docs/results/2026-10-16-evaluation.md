# MedRAG evaluation — 2026-10-16

Generator: `openai/gpt-oss-120b` · Judge: `qwen/qwen3.8-27b`

Gold items: in_scope 89, out_of_scope 22, near_miss 12, adversarial 6

## Answer quality

| system | in-scope n | answered | graded | correct (of graded) | correct or partial | correct (overall) | book support | citation support | cited gold page | OOS refused | refusal precision | near-miss ok | adversarial ok | judge errors | LLM errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medrag | 89 | 100.0% | 89 | 92.1% | 100.0% | 92.1% | 98.2% | 97.4% | 98.9% | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 0 |

*Correct (overall)*: correct answers over in-scope items that were refused or graded (items whose judge failed are excluded and counted under judge errors). *Book support*: mean, over answered items, of the share of an answer's claims stated by the retrieved textbook passages (the same passages for both systems). *Citation support*: share of cited claims supported by the passages they cite. *Cited gold page*: answered items citing at least one gold page. *Refusal precision*: share of refusals that were for questions the book does not answer. *OOS refused* counts only explicit refusals.

## Retrieval (in-scope items; metrics over the top-k unique pages)

| variant | n | recall@1 | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| dense | 89 | 80.9% | 95.5% | 95.5% | 0.868 | 0.876 |
| bm25 | 89 | 73.0% | 91.0% | 96.6% | 0.818 | 0.824 |
| v1_merge | 89 | 19.1% | 31.5% | 37.1% | 0.242 | 0.242 |
| rrf | 89 | 91.0% | 98.9% | 98.9% | 0.939 | 0.945 |
| rrf_rerank | 89 | 98.9% | 100.0% | 100.0% | 0.994 | 0.995 |

## Refusal threshold calibration

Chosen threshold (dev, in-scope pass ≥ 98.0%): **0.0568** (current setting 0.0450)

| split | in-scope passed | out-of-scope rejected |
|---|---|---|
| dev | 100.0% | 100.0% |
| test | 100.0% | 100.0% |

## Consistency

20 questions × 3 runs. Answer text cosine: **0.965** · cited-page Jaccard: **0.972**

