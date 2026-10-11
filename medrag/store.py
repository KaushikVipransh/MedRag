from pathlib import Path

from medrag.types import Chunk


class ChunkStore:
    def __init__(self, path: Path, collection: str):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._col.count()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]], batch: int = 500) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self._col.add(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=embeddings[i : i + batch],
                metadatas=[c.metadata() for c in part],
            )

    def delete_document(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def query(
        self, embedding: list[float], n: int, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        n = min(n, self.count())
        if n == 0:
            return []
        where = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (Chunk.from_record(i, d, m), 1.0 - dist)
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def all_chunks(self) -> list[Chunk]:
        res = self._col.get(include=["documents", "metadatas"])
        return [
            Chunk.from_record(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def documents(self) -> dict[str, str]:
        res = self._col.get(include=["metadatas"])
        return {m["doc_id"]: m["doc_title"] for m in res["metadatas"]}
