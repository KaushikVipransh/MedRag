from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawDocument:
    pages_lines: list[list[str]]
    pdf_labels: list[str]
    toc: list[list]
    title: str


def read_pdf(path: Path) -> RawDocument:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return RawDocument(
            pages_lines=[page.get_text("text").splitlines() for page in doc],
            pdf_labels=[page.get_label() for page in doc],
            toc=doc.get_toc(simple=True),
            title=(doc.metadata or {}).get("title") or "",
        )


def read_font_lines(path: Path) -> list[list[tuple[str, float]]]:
    import pymupdf

    out: list[list[tuple[str, float]]] = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = line["spans"]
                    text = "".join(s["text"] for s in spans).strip()
                    if text:
                        lines.append((text, max(s["size"] for s in spans)))
            out.append(lines)
    return out
