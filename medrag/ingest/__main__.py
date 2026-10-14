import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from medrag.config import Settings
from medrag.embeddings import SentenceTransformerEmbedder
from medrag.ingest.build import ingest_pdf
from medrag.store import ChunkStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF(s) into the MedRAG store.")
    parser.add_argument("source", type=Path, help="a PDF file or a directory of PDFs")
    parser.add_argument("--title", help="display title (only with a single file)")
    args = parser.parse_args()

    pdfs = sorted(args.source.glob("*.pdf")) if args.source.is_dir() else [args.source]
    if not pdfs:
        raise SystemExit(f"No PDFs found at {args.source}")
    if args.title and len(pdfs) > 1:
        raise SystemExit("--title can only be used with a single PDF")

    settings = Settings.from_env()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    store = ChunkStore(settings.chroma_dir, settings.collection)
    embedder = SentenceTransformerEmbedder(settings.embed_model)

    manifest_path = settings.chroma_dir / f"manifest-{settings.collection}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for pdf in pdfs:
        started = time.time()
        stats = ingest_pdf(
            pdf, store, embedder, settings.chunk_max_tokens, settings.chunk_overlap_tokens, args.title
        )
        print(
            f"{stats.doc_id}: {stats.chunks} chunks from {stats.pages_indexed}/{stats.pages_total} pages "
            f"in {time.time() - started:.0f}s"
        )
        if stats.pages_without_text > 0.2 * stats.pages_total:
            print(f"  WARNING: {stats.pages_without_text} pages have no text layer (scanned?). OCR them first.")
        manifest[stats.doc_id] = {
            **asdict(stats),
            "source": str(pdf),
            "embed_model": settings.embed_model,
            "chunk_max_tokens": settings.chunk_max_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Index now holds {store.count()} chunks in {settings.chroma_dir}")


if __name__ == "__main__":
    main()
