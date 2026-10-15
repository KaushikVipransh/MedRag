"""Chunks as Postgres rows, and an in-memory vector store over one book's chunks. A book is
loaded from the database once and searched here; dense search is brute-force cosine, which
takes milliseconds for a few thousand chunks."""
import numpy as np

from medrag.types import Chunk


def chunk_row(chunk: Chunk, embedding: list[float], user_id: str) -> dict:
    return {
        "id": chunk.id,
        "book_id": chunk.doc_id,
        "user_id": user_id,
        "text": chunk.text,
        "page_index": chunk.page_index,
        "page_label": chunk.page_label,
        "section": chunk.section,
        "topic": chunk.topic,
        "embedding": [round(float(x), 6) for x in embedding],
    }


def row_chunk(row: dict, doc_title: str) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["book_id"],
        doc_title=doc_title,
        text=row["text"],
        page_index=int(row["page_index"]),
        page_label=str(row["page_label"]),
        section=row.get("section") or "",
        topic=row.get("topic") or "",
    )


class InMemoryChunkStore:
    """Read-only stand-in for ChunkStore with the methods retrieval uses."""

    def __init__(self, chunks: list[Chunk], embeddings: list[list[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings differ in length")
        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    @classmethod
    def from_rows(cls, rows: list[dict], doc_title: str) -> "InMemoryChunkStore":
        return cls([row_chunk(r, doc_title) for r in rows], [r["embedding"] for r in rows])

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def documents(self) -> dict[str, str]:
        return {c.doc_id: c.doc_title for c in self._chunks}

    def query(
        self, embedding: list[float], n: int, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        if not self._chunks or n <= 0:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._matrix @ q
        hits: list[tuple[Chunk, float]] = []
        for i in np.argsort(-scores, kind="stable"):
            chunk = self._chunks[i]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            hits.append((chunk, float(scores[i])))
            if len(hits) == n:
                break
        return hits
