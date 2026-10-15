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
from medrag.ingest.topics import generic_headings, pretty, raw_headings
from medrag.store import ChunkStore
from medrag.types import Chunk, context_text


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
    bodies = [strip_margin_lines(lines, repeating) for lines in pages_lines]
    generic = generic_headings(bodies)
    chunks: list[Chunk] = []
    pages_indexed = 0
    pages_without_text = 0
    topic, topic_section = "", None
    for i, body in enumerate(bodies):
        if sections[i] != topic_section:  # topics do not carry over chapter boundaries
            topic, topic_section = "", sections[i]
        page_start_topic = topic
        heads = [h for h in raw_headings(body) if h not in generic]
        if heads:
            topic = pretty(heads[-1])
        text = lines_to_text(body)
        if len(text) < 100:
            pages_without_text += 1
            continue
        if is_non_content_page(body):
            continue
        pieces = [p for p in chunk_text(text, count_tokens, max_tokens, overlap_tokens) if len(p) >= min_chars]
        if pieces:
            pages_indexed += 1
        flat = " ".join(text.split())
        marks, cursor = [], 0
        for h in heads:
            pos = flat.find(h, cursor)
            if pos >= 0:
                marks.append((pos, pretty(h)))
                cursor = pos + len(h)
        cursor = 0
        for n, piece in enumerate(pieces):
            start = flat.find(" ".join(piece.split())[:60], cursor)
            start = cursor if start < 0 else start
            cursor = start
            chunk_topic = page_start_topic
            for pos, name in marks:
                if pos <= start:
                    chunk_topic = name
            chunks.append(
                Chunk(
                    id=f"{doc_id}-p{i:04d}-c{n:03d}",
                    doc_id=doc_id,
                    doc_title=title,
                    text=piece,
                    page_index=i,
                    page_label=labels[i],
                    section=sections[i],
                    topic=chunk_topic,
                )
            )
    return chunks, pages_indexed, pages_without_text


def chunks_for_pdf(
    source: Path | bytes,
    doc_id: str,
    title: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[list[Chunk], IngestStats]:
    raw = read_pdf(source)
    n_pages = len(raw.pages_lines)
    sections = (
        sections_from_toc(raw.toc, n_pages) if raw.toc else sections_from_fonts(read_font_lines(source))
    )
    chunks, pages_indexed, pages_without_text = build_chunks(
        doc_id, title, raw.pages_lines, raw.pdf_labels, sections, count_tokens, max_tokens, overlap_tokens,
    )
    return chunks, IngestStats(
        doc_id=doc_id, title=title, pages_total=n_pages, pages_indexed=pages_indexed,
        pages_without_text=pages_without_text, chunks=len(chunks),
    )


def ingest_pdf(
    path: Path,
    store: ChunkStore,
    embedder: Embedder,
    max_tokens: int,
    overlap_tokens: int,
    title: str | None = None,
) -> IngestStats:
    path = Path(path)
    doc_id = slugify(path.stem)
    title = title or path.stem.replace("_", " ").replace("-", " ").title()
    chunks, stats = chunks_for_pdf(path, doc_id, title, embedder.count_tokens, max_tokens, overlap_tokens)
    store.delete_document(doc_id)
    if chunks:
        store.add(chunks, embedder.encode([context_text(c) for c in chunks]))
    return stats
