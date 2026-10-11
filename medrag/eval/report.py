from statistics import median

from medrag.eval.metrics import mean


def _share(flags) -> float | None:
    flags = [bool(f) for f in flags]
    return sum(flags) / len(flags) if flags else None


def summarize_system(results) -> dict:
    errors = [r for r in results if "system llm_error" in r.judge_errors]
    valid = [r for r in results if r not in errors]
    ins = [r for r in valid if r.category == "in_scope"]
    answered = [r for r in ins if not r.refused]
    graded = [r for r in answered if r.correctness is not None]

    def handled(cat):
        return _share(r.handled_ok for r in valid if r.category == cat and r.handled_ok is not None)

    return {
        "n_in_scope": len(ins),
        "answer_rate": _share(not r.refused for r in ins),
        "correct": _share(r.correctness == "correct" for r in graded),
        "correct_or_partial": _share(r.correctness in ("correct", "partial") for r in graded),
        "correct_overall": (sum(r.correctness == "correct" for r in ins) / len(ins)) if ins else None,
        "book_support": mean(r.book_support for r in answered),
        "citation_support": mean(r.citation_support for r in answered),
        "oos_refusal": handled("out_of_scope"),
        "near_miss_ok": handled("near_miss"),
        "adversarial_ok": handled("adversarial"),
        "median_latency_s": median([r.latency_s for r in ins]) if ins else None,
        "llm_errors": len(errors),
    }


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _num(v, digits=3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def render_report(meta, systems=None, retrieval=None, calibration=None, consistency=None) -> str:
    lines = []
    if meta.get("unverified_included"):
        lines += ["> **DRAFT** — this run includes gold items that have not been verified yet. "
                  "Do not cite these numbers.", ""]
    lines += [f"# MedRAG evaluation — {meta.get('date', '')}", "",
              f"Generator: `{meta.get('generator')}` · Judge: `{meta.get('judge')}`", "",
              "Gold items: " + ", ".join(f"{k} {v}" for k, v in meta.get("gold_counts", {}).items()), ""]

    if systems:
        lines += ["## Answer quality", "",
                  "| system | in-scope n | answered | correct (of answered) | correct or partial | correct (overall) "
                  "| book support | citation support | OOS refused | near-miss ok | adversarial ok | median latency |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, s in systems.items():
            lines.append(
                f"| {name} | {s['n_in_scope']} | {_pct(s['answer_rate'])} | {_pct(s['correct'])} "
                f"| {_pct(s['correct_or_partial'])} | {_pct(s['correct_overall'])} | {_pct(s['book_support'])} "
                f"| {_pct(s['citation_support'])} | {_pct(s['oos_refusal'])} | {_pct(s['near_miss_ok'])} "
                f"| {_pct(s['adversarial_ok'])} | {_num(s['median_latency_s'], 1)} s |"
            )
        lines += ["", "*Book support*: share of answer claims stated by the retrieved textbook passages "
                  "(the same passages for both systems). *Citation support*: share of cited claims "
                  "supported by the passages they cite.", ""]

    if retrieval:
        lines += ["## Retrieval (page-level, in-scope items)", "",
                  "| variant | n | recall@1 | recall@5 | recall@10 | MRR | nDCG@5 |", "|---|---|---|---|---|---|---|"]
        for name, m in retrieval.items():
            lines.append(f"| {name} | {m['n']} | {_pct(m['recall@1'])} | {_pct(m['recall@5'])} "
                         f"| {_pct(m['recall@10'])} | {_num(m['mrr'])} | {_num(m['ndcg@5'])} |")
        lines.append("")

    if calibration:
        lines += ["## Refusal threshold calibration", "",
                  f"Chosen threshold (dev, in-scope pass ≥ {_pct(calibration['min_pass'])}): "
                  f"**{_num(calibration['threshold'], 4)}** (current setting {_num(calibration['current'], 4)})", "",
                  "| split | in-scope passed | out-of-scope rejected |", "|---|---|---|",
                  f"| dev | {_pct(calibration['dev'][0])} | {_pct(calibration['dev'][1])} |",
                  f"| test | {_pct(calibration['test'][0])} | {_pct(calibration['test'][1])} |", ""]

    if consistency:
        lines += ["## Consistency", "",
                  f"{consistency['n_questions']} questions × {consistency['repeats']} runs. "
                  f"Answer text cosine: **{_num(consistency['text_cosine'])}** · "
                  f"cited-page Jaccard: **{_num(consistency['citation_jaccard'])}**", ""]
    return "\n".join(lines)
