"""Run the real pipeline on known questions and print what a reviewer needs to judge it."""
import time

from medrag.config import Settings
from medrag.pipeline import build_pipeline

QUESTIONS = [
    ("in", "What is Ménière's disease?"),
    ("in", "What is tinnitus retraining therapy?"),
    ("in", "What are the orbital complications of sinusitis?"),
    ("in", "What is FESS (functional endoscopic sinus surgery)?"),
    ("in", "What are the parts of the nasal septum?"),
    ("out", "What is the recommended COVID-19 vaccine schedule for 2025?"),
    ("out", "How do ACE inhibitors lower blood pressure?"),
    ("out", "What are the symptoms of appendicitis?"),
]

rag = build_pipeline(Settings.from_env())
for scope, q in QUESTIONS:
    t = time.time()
    a = rag.answer(q)
    top = a.chunks[0].rerank_score if a.chunks else None
    print("=" * 100)
    print(f"[{scope}] {q}   ({time.time() - t:.1f}s, top rerank={top if top is None else round(top, 4)})")
    print(f"refused={a.refused} reason={a.refusal_reason} warnings={a.warnings}")
    print(a.text)
    for c in a.citations:
        print(f"   [{c.marker}] p.{c.chunk.page_label} {c.chunk.section} :: {c.chunk.text[:120]!r}")
