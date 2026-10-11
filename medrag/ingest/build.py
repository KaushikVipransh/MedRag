import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from medrag.embeddings import Embedder
from medrag.ingest.chunking import chunk_text
from medrag.ingest.clean import (
    find_repeating_margin_lines,
    is_non_content_page,
    lines_to_text,
    strip_margin_lines,
)
from medrag.ingest.labels import infer_page_labels
from medrag.ingest.pdf import read_font_lines, read_pdf
from medrag.ingest.sections import sections_from_fonts, sections_from_toc
from medrag.store import ChunkStore
from medrag.types import Chunk


@dataclass
class IngestStats:
    doc_id: str
    title: str
    pages_total: int
    pages_indexed: int
    pages_without_text: int
    chunks: int


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_chunks(
    doc_id: str,
    title: str,
    pages_lines: list[list[str]],
    pdf_labels: list[str],
    sections: list[str],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int,
    min_chars: int = 40,
) -> tuple[list[Chunk], int, int]:
    """Returns (chunks, pages_indexed, pages_without_text). A page is "without text"
    when nothing usable remains after removing headers, footers and watermarks, which
    is how image-only (scanned) pages look even when they carry a text overlay."""
    labels = infer_page_labels(pages_lines, pdf_labels)
    repeating = find_repeating_margin_lines(pages_lines)
    chunks: list[Chunk] = []
    pages_indexed = 0
    pages_without_text = 0
    for i, lines in enumerate(pages_lines):
        body = strip_margin_lines(lines, repeating)
        text = lines_to_text(body)
        if len(text) < 100:
            pages_without_text += 1
            continue
        if is_non_content_page(body):
            continue
        pieces = [p for p in chunk_text(text, count_tokens, max_tokens, overlap_tokens) if len(p) >= min_chars]
        if pieces:
            pages_indexed += 1
        for n, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{doc_id}-p{i:04d}-c{n:03d}",
                    doc_id=doc_id,
                    doc_title=title,
                    text=piece,
                    page_index=i,
                    page_label=labels[i],
                    section=sections[i],
                )
            )
    return chunks, pages_indexed, pages_without_text


def ingest_pdf(
    path: Path,
    store: ChunkStore,
    embedder: Embedder,
    max_tokens: int,
    overlap_tokens: int,
    title: str | None = None,
) -> IngestStats:
    path = Path(path)
    raw = read_pdf(path)
    doc_id = slugify(path.stem)
    title = title or path.stem.replace("_", " ").replace("-", " ").title()
    n_pages = len(raw.pages_lines)
    sections = (
        sections_from_toc(raw.toc, n_pages) if raw.toc else sections_from_fonts(read_font_lines(path))
    )
    chunks, pages_indexed, pages_without_text = build_chunks(
        doc_id, title, raw.pages_lines, raw.pdf_labels, sections,
        embedder.count_tokens, max_tokens, overlap_tokens,
    )
    store.delete_document(doc_id)
    if chunks:
        store.add(chunks, embedder.encode([c.text for c in chunks]))
    return IngestStats(
        doc_id=doc_id,
        title=title,
        pages_total=n_pages,
        pages_indexed=pages_indexed,
        pages_without_text=pages_without_text,
        chunks=len(chunks),
    )
