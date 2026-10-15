"""Helpers for multi-user tests: small generated PDFs (more helpers are added in Task 11)."""
import pymupdf

BODIES = [
    "Tinnitus is ringing in the ear.\nIt may be subjective or objective.\nMaskers help some patients.\nRetraining aims at habituation.",
    "Otosclerosis causes conductive deafness.\nStapedectomy is the usual operation.\nHearing aids are an option.\nFluoride may slow it.",
    "The nasal septum has three parts.\nThese are columellar, membranous\nand septum proper.\nDeviation causes obstruction.",
    "Acute sinusitis follows a cold.\nMaxillary sinus is most affected.\nTreatment uses antibiotics.\nDecongestants aid drainage.",
]


def make_pdf_bytes(bodies: list[str], blank=(), watermark: bool = False) -> bytes:
    """One page per body with a running header and a printed page number from 101.
    Pages listed in `blank` get no body: only the header, number and optional watermark,
    which is what an image-only (scanned) page looks like to text extraction."""
    doc = pymupdf.open()
    for n, body in enumerate(bodies):
        page = doc.new_page()
        page.insert_text((72, 60), "Test Book of ENT")
        if n not in blank:
            page.insert_text((72, 120), body)
        page.insert_text((300, 800), str(n + 101))
        if watermark:
            page.insert_text((72, 820), "mebooksfree.com")
    data = doc.tobytes()
    doc.close()
    return data
