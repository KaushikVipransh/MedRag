from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    text: str
    page_index: int  # 0-based index in the PDF
    page_label: str  # page number as printed in the book
    section: str  # "Chapter > Section", "" if unknown

    def metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "page_index": self.page_index,
            "page_label": self.page_label,
            "section": self.section,
        }

    @classmethod
    def from_record(cls, id: str, text: str, meta: dict) -> "Chunk":
        return cls(
            id=id,
            doc_id=meta["doc_id"],
            doc_title=meta["doc_title"],
            text=text,
            page_index=int(meta["page_index"]),
            page_label=str(meta["page_label"]),
            section=meta.get("section", ""),
        )


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # value used for final ordering
    rrf_score: float = 0.0
    rerank_score: float | None = None


@dataclass(frozen=True)
class Citation:
    marker: int  # the n in [n]
    chunk: Chunk


@dataclass
class Answer:
    text: str
    refused: bool
    refusal_reason: str | None = None
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    standalone_question: str = ""
    emergency: bool = False
    warnings: list[str] = field(default_factory=list)
