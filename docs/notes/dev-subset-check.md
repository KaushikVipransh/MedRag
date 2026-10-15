# Dev-subset check — relevance threshold 0.045

**Date:** 2026-10-15
**Threshold:** `rerank_threshold = 0.045` (was 0.06; weakest verified in-scope 0.057, strongest out-of-scope 0.037)

## Commands

```bash
docker compose run --rm dev python -m medrag.eval run --systems medrag --ids-file data/eval/dev_subset.txt
docker compose run --rm dev python -m medrag.eval report
```

The run finished on 2026-10-15 (no `QuotaExhausted`; one Docker restart mid-run, resumed from the LLM cache).

## Result (40 items: in-scope 24, out-of-scope 6, near-miss 6, adversarial 4)

| system | in-scope n | answered | graded | correct (of graded) | correct or partial | book support | citation support | cited gold page | OOS refused | refusal precision | near-miss ok | adversarial ok | judge errors | LLM errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medrag | 24 | 100.0% | 24 | 87.5% | 100.0% | 96.8% | 95.4% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 0 |

## Reading

- Out-of-scope, near-miss and adversarial handling: 100% — meets the baseline.
- In-scope: 21 correct, 3 partial (in-030, in-036, in-042), 0 incorrect, none refused. 87.5% is below the 129-question
  baseline of 90.5%, but on 24 items one answer is ~4 points, and the threshold only decides refusals: every in-scope
  item was answered, so the partials are answer-quality/judge variance, not a gate effect.
