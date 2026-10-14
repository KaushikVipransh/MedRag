import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from medrag.config import Settings
from medrag.eval.calibrate import choose_threshold, rates, split_dev_test
from medrag.eval.config import EvalSettings
from medrag.eval.consistency import consistency_scores
from medrag.eval.gold import load_gold, save_evidence, save_gold, select
from medrag.eval.report import paired_book_support, render_report, summarize_system

UNVERIFIED_HELP = (
    "No verified gold items. Verify them with: streamlit run review_gold.py "
    "(or pass --include-unverified for a DRAFT dry run)."
)


def _cached_llm(settings: Settings, cfg: EvalSettings, model: str):
    from medrag.eval.llm_wrappers import CachedLLM, PatientLLM
    from medrag.generation.llm import GroqClient

    return CachedLLM(PatientLLM(GroqClient(settings.groq_api_key, model)), model, cfg.cache_dir)


def _items(cfg: EvalSettings, args, categories=None, limit=None):
    items = select(load_gold(cfg.gold_path), args.include_unverified, categories, limit)
    if not items:
        raise SystemExit(UNVERIFIED_HELP)
    return items


def _meta(settings, cfg, items, args) -> dict:
    return {
        "date": time.strftime("%Y-%m-%d"),
        "generator": settings.llm_model,
        "judge": cfg.judge_model,
        "unverified_included": any(not i.verified for i in items),
        "gold_counts": dict(Counter(i.category for i in items)),
    }


def _write_json(cfg: EvalSettings, name: str, payload: dict) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_draft(args, settings, cfg):
    if cfg.gold_path.exists() and not args.force:
        raise SystemExit(f"{cfg.gold_path} already exists (it may hold reviewed work). Pass --force to overwrite.")
    from medrag.eval.draft import draft_gold
    from medrag.eval.fixed_sets import fixed_items
    from medrag.store import ChunkStore

    chunks = ChunkStore(settings.chroma_dir, settings.collection).all_chunks()
    items, evidence, stats = draft_gold(_cached_llm(settings, cfg, cfg.drafter_model), chunks, args.n)
    save_gold(cfg.gold_path, items + fixed_items())
    save_evidence(cfg.evidence_path, evidence)
    print(f"Drafted {len(items)} in-scope items ({dict(stats)}) + {len(fixed_items())} fixed items -> {cfg.gold_path}")


def cmd_retrieval(args, settings, cfg):
    items = _items(cfg, args, {"in_scope"})
    from medrag.embeddings import SentenceTransformerEmbedder
    from medrag.eval.metrics import score_retriever
    from medrag.eval.variants import build_variants
    from medrag.retrieval.bm25 import BM25Index
    from medrag.retrieval.rerank import CrossEncoderReranker
    from medrag.retrieval.retriever import HybridRetriever
    from medrag.store import ChunkStore

    embedder = SentenceTransformerEmbedder(settings.embed_model)
    reranker = CrossEncoderReranker(settings.rerank_model)
    rows = {}
    for n, collection in enumerate(args.collections.split(",")):
        store = ChunkStore(settings.chroma_dir, collection)
        if store.count() == 0:
            print(f"skip {collection}: empty")
            continue
        chunks = store.all_chunks()
        bm25 = BM25Index(chunks)
        if n == 0:
            variants = build_variants(store, embedder, bm25, chunks, reranker, top_n=10)
        else:
            variants = {f"rrf_rerank[{collection}]": HybridRetriever(store, embedder, bm25, reranker, top_n=10)}
        for name, retriever in variants.items():
            rows[name] = score_retriever(retriever, items)
            print(name, rows[name])
    _write_json(cfg, "retrieval.json", {"meta": _meta(settings, cfg, items, args), "variants": rows})


def cmd_run(args, settings, cfg):
    items = _items(cfg, args, limit=args.limit)
    from medrag.eval.baseline import NoRagBaseline
    from medrag.eval.runner import run_system
    from medrag.pipeline import build_pipeline

    gen = _cached_llm(settings, cfg, settings.llm_model)
    judge = _cached_llm(settings, cfg, cfg.judge_model)
    rag = build_pipeline(settings, llm=gen)
    memo: dict[str, list] = {}

    def passages_for(question):
        if question not in memo:
            memo[question] = rag.retriever.retrieve(question)
        return memo[question]

    systems = {"medrag": rag, "no_rag": NoRagBaseline(gen, settings.temperature)}
    config = (f"generator={settings.llm_model};judge={cfg.judge_model};threshold={settings.rerank_threshold};"
              f"top_n={settings.top_n};candidates={settings.candidates_per_retriever};"
              f"rerank={settings.rerank_model};gate={settings.gate_model}")
    summaries, all_results = {}, {}
    for name in args.systems.split(","):
        results = run_system(items, name, systems[name], judge, passages_for,
                             cfg.results_dir / f"{name}.jsonl", config=config)
        all_results[name] = results
        summaries[name] = summarize_system(results)
    paired = None
    if {"medrag", "no_rag"} <= set(all_results):
        a, b, n = paired_book_support(all_results["medrag"], all_results["no_rag"])
        paired = {"medrag": a, "no_rag": b, "n": n}
    _write_json(cfg, "summary.json",
                {"meta": _meta(settings, cfg, items, args), "systems": summaries, "paired_book_support": paired})
    print(json.dumps(summaries, indent=2))


def cmd_calibrate(args, settings, cfg):
    items = _items(cfg, args, {"in_scope", "out_of_scope"})
    from medrag.pipeline import build_pipeline

    retriever = build_pipeline(settings).retriever
    score = {i.id: (retriever.retrieve_with_relevance(i.question)[1] or 0.0) for i in items}
    dev, test = split_dev_test(items)

    def split_scores(group):
        return ([score[i.id] for i in group if i.category == "in_scope"],
                [score[i.id] for i in group if i.category == "out_of_scope"])

    threshold = choose_threshold(*split_scores(dev), min_pass=args.min_pass)
    payload = {
        "threshold": threshold, "current": settings.rerank_threshold, "min_pass": args.min_pass,
        "dev": rates(*split_scores(dev), threshold), "test": rates(*split_scores(test), threshold),
        "scores": score,
    }
    _write_json(cfg, "calibration.json", {"meta": _meta(settings, cfg, items, args), "calibration": payload})
    print({k: v for k, v in payload.items() if k != "scores"})


def cmd_consistency(args, settings, cfg):
    items = _items(cfg, args, {"in_scope"})[: args.limit]
    from medrag.pipeline import build_pipeline

    gen = _cached_llm(settings, cfg, settings.llm_model)
    rag = build_pipeline(settings, llm=gen)
    runs = []
    for item in items:
        answers = []
        for rep in range(args.repeats):
            gen.salt = f"consistency-rep{rep}"
            answers.append(rag.answer(item.question))
        runs.append(answers)
    gen.salt = ""
    scores = consistency_scores(runs, rag.retriever.embedder.encode)
    payload = {**scores, "n_questions": len(items), "repeats": args.repeats}
    _write_json(cfg, "consistency.json", {"meta": _meta(settings, cfg, items, args), "consistency": payload})
    print({k: v for k, v in payload.items() if k != "per_question"})


def cmd_report(args, settings, cfg):
    loaded = {}
    for name in ("summary", "retrieval", "calibration", "consistency"):
        path = cfg.results_dir / f"{name}.json"
        if path.exists():
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    if not loaded:
        raise SystemExit(f"No results in {cfg.results_dir}. Run retrieval/run/calibrate/consistency first.")
    metas = [v["meta"] for v in loaded.values()]
    meta = {**metas[0], "unverified_included": any(m.get("unverified_included") for m in metas)}
    md = render_report(
        meta,
        systems=loaded.get("summary", {}).get("systems"),
        retrieval=loaded.get("retrieval", {}).get("variants"),
        calibration=loaded.get("calibration", {}).get("calibration"),
        consistency=loaded.get("consistency", {}).get("consistency"),
        paired=loaded.get("summary", {}).get("paired_book_support"),
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-DRAFT" if meta["unverified_included"] else ""
    out = cfg.report_dir / f"{meta['date']}-evaluation{suffix}.md"
    out.write_text(md + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def cmd_all(args, settings, cfg):
    """Run every stage in order with default options, then write the report."""
    flag = args.include_unverified
    stages = [
        (cmd_retrieval, argparse.Namespace(include_unverified=flag, collections=args.collections)),
        (cmd_calibrate, argparse.Namespace(include_unverified=flag, min_pass=0.98)),
        (cmd_run, argparse.Namespace(include_unverified=flag, systems="medrag,no_rag", limit=None)),
        (cmd_consistency, argparse.Namespace(include_unverified=flag, limit=20, repeats=3)),
        (cmd_report, argparse.Namespace()),
    ]
    for stage, stage_args in stages:
        stage(stage_args, settings, cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m medrag.eval")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("draft", help="draft in-scope gold items from the index")
    p.add_argument("--n", type=int, default=90)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("retrieval", help="retrieval metrics for ablation variants")
    p.add_argument("--include-unverified", action="store_true")
    p.add_argument("--collections", default="medrag")
    p = sub.add_parser("run", help="answer-quality evaluation with LLM judges")
    p.add_argument("--include-unverified", action="store_true")
    p.add_argument("--systems", default="medrag,no_rag")
    p.add_argument("--limit", type=int, default=None, help="max items per category")
    p = sub.add_parser("calibrate", help="calibrate the refusal threshold")
    p.add_argument("--include-unverified", action="store_true")
    p.add_argument("--min-pass", type=float, default=0.98)
    p = sub.add_parser("consistency", help="repeat answers and measure stability")
    p.add_argument("--include-unverified", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    sub.add_parser("report", help="write the markdown report from saved results")
    p = sub.add_parser("all", help="retrieval, calibrate, run, consistency, then report")
    p.add_argument("--include-unverified", action="store_true")
    p.add_argument("--collections", default="medrag,medrag_t128,medrag_t300")

    args = parser.parse_args(argv)
    settings, cfg = Settings.from_env(), EvalSettings.from_env()
    {
        "draft": cmd_draft, "retrieval": cmd_retrieval, "run": cmd_run,
        "calibrate": cmd_calibrate, "consistency": cmd_consistency, "report": cmd_report, "all": cmd_all,
    }[args.command](args, settings, cfg)


if __name__ == "__main__":
    main()
