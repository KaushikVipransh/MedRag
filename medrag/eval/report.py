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
    decided = [r for r in ins if r.refused or r.correctness is not None]  # judge errors excluded
    with_gold = [r for r in answered if r.gold_pages]
    refusals = [r for r in valid if r.refused]

    def handled(cat):
        return _share(r.handled_ok for r in valid if r.category == cat and r.handled_ok is not None)

    return {
        "n_in_scope": len(ins),
        "answer_rate": _share(not r.refused for r in ins),
        "correct": _share(r.correctness == "correct" for r in graded),
        "correct_or_partial": _share(r.correctness in ("correct", "partial") for r in graded),
        "correct_overall": _share(r.correctness == "correct" for r in decided),
        "graded": len(graded),
        "judge_errors": sum(bool(r.judge_errors) for r in valid),
        "citation_hit": _share(set(r.cited_pages) & set(r.gold_pages) for r in with_gold),
        "refusal_precision": _share(r.category != "in_scope" for r in refusals),
        "book_support": mean(r.book_support for r in answered),
        "citation_support": mean(r.citation_support for r in answered),
        "oos_refusal": handled("out_of_scope"),
        "near_miss_ok": handled("near_miss"),
        "adversarial_ok": handled("adversarial"),
        "median_latency_s": median([r.latency_s for r in ins]) if ins else None,
        "llm_errors": len(errors),
    }


def paired_book_support(a, b) -> tuple[float | None, float | None, int]:
    """Book support of two systems on the in-scope items both answered and both had judged."""
    bs = {r.id: r.book_support for r in b if r.category == "in_scope" and r.book_support is not None}
    pairs = [(r.book_support, bs[r.id]) for r in a
             if r.category == "in_scope" and r.book_support is not None and r.id in bs]
    return mean(x for x, _ in pairs), mean(y for _, y in pairs), len(pairs)


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _num(v, digits=3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def render_report(meta, systems=None, retrieval=None, calibration=None, consistency=None, paired=None) -> str:
    lines = []
    if meta.get("unverified_included"):
        lines += ["> **DRAFT** — this run includes gold items that have not been verified yet. "
                  "Do not cite these numbers.", ""]
    lines += [f"# MedRAG evaluation — {meta.get('date', '')}", "",
              f"Generator: `{meta.get('generator')}` · Judge: `{meta.get('judge')}`", "",
              "Gold items: " + ", ".join(f"{k} {v}" for k, v in meta.get("gold_counts", {}).items()), ""]

    if systems:
        lines += ["## Answer quality", "",
                  "| system | in-scope n | answered | graded | correct (of graded) | correct or partial "
                  "| correct (overall) | book support | citation support | cited gold page | OOS refused "
                  "| refusal precision | near-miss ok | adversarial ok | judge errors | LLM errors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, s in systems.items():
            lines.append(
                f"| {name} | {s['n_in_scope']} | {_pct(s['answer_rate'])} | {s['graded']} | {_pct(s['correct'])} "
                f"| {_pct(s['correct_or_partial'])} | {_pct(s['correct_overall'])} | {_pct(s['book_support'])} "
                f"| {_pct(s['citation_support'])} | {_pct(s['citation_hit'])} | {_pct(s['oos_refusal'])} "
                f"| {_pct(s['refusal_precision'])} | {_pct(s['near_miss_ok'])} | {_pct(s['adversarial_ok'])} "
                f"| {s['judge_errors']} | {s['llm_errors']} |"
            )
        lines += ["",
                  "*Correct (overall)*: correct answers over in-scope items that were refused or graded "
                  "(items whose judge failed are excluded and counted under judge errors). "
                  "*Book support*: mean, over answered items, of the share of an answer's claims stated by "
                  "the retrieved textbook passages (the same passages for both systems). "
                  "*Citation support*: share of cited claims supported by the passages they cite. "
                  "*Cited gold page*: answered items citing at least one gold page. "
                  "*Refusal precision*: share of refusals that were for questions the book does not answer. "
                  "*OOS refused* counts only explicit refusals.", ""]
        if paired and paired.get("n"):
            lines += [f"Paired book support on the {paired['n']} in-scope items both systems answered: "
                      f"medrag **{_pct(paired.get('medrag'))}** vs no_rag **{_pct(paired.get('no_rag'))}**.", ""]

    if retrieval:
        lines += ["## Retrieval (in-scope items; metrics over the top-k unique pages)", "",
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
