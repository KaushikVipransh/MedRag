"""Checks an upload before anything is stored: it opens as a PDF, is within the size and
page limits, and has a text layer. Scanned books are rejected rather than OCR'd."""
from dataclasses import dataclass

from medrag.ingest.clean import find_repeating_margin_lines, lines_to_text, strip_margin_lines

NOT_A_PDF_MESSAGE = "This file couldn't be opened as a PDF."
SCANNED_MESSAGE = "This PDF has no text layer (it looks scanned). Upload a text-based PDF."
MIN_PAGE_CHARS = 100  # same rule ingestion uses for "a page without text"
MAX_NO_TEXT_SHARE = 0.2


class PdfRejected(ValueError):
    """str(error) is safe to show to the user."""


@dataclass(frozen=True)
class PdfCheck:
    pages: int
    text_pages: int


def check_pdf(data: bytes, max_bytes: int, max_pages: int) -> PdfCheck:
    import pymupdf

    if len(data) > max_bytes:
        mb = 1024 * 1024
        raise PdfRejected(f"This PDF is {len(data) / mb:.1f} MB; the limit is {max_bytes // mb} MB.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PdfRejected(NOT_A_PDF_MESSAGE) from exc
    with doc:
        if doc.needs_pass or doc.page_count == 0:
            raise PdfRejected(NOT_A_PDF_MESSAGE)
        if doc.page_count > max_pages:
            raise PdfRejected(f"This PDF has {doc.page_count} pages; the limit is {max_pages}.")
        try:
            pages_lines = [page.get_text("text").splitlines() for page in doc]
        except Exception as exc:
            raise PdfRejected(NOT_A_PDF_MESSAGE) from exc
    repeating = find_repeating_margin_lines(pages_lines)
    text_pages = sum(
        1 for lines in pages_lines if len(lines_to_text(strip_margin_lines(lines, repeating))) >= MIN_PAGE_CHARS
    )
    pages = len(pages_lines)
    if pages - text_pages >= MAX_NO_TEXT_SHARE * pages:
        raise PdfRejected(SCANNED_MESSAGE)
    return PdfCheck(pages=pages, text_pages=text_pages)
