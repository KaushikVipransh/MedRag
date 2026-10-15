import pymupdf
import pytest

from medrag.ingest.precheck import NOT_A_PDF_MESSAGE, SCANNED_MESSAGE, PdfRejected, check_pdf
from tests.fakes_cloud import BODIES, make_pdf_bytes

MB = 1024 * 1024
TEN = (BODIES * 3)[:10]


def check(data, max_bytes=50 * MB, max_pages=1000):
    return check_pdf(data, max_bytes=max_bytes, max_pages=max_pages)


def test_text_pdf_passes():
    result = check(make_pdf_bytes(BODIES))
    assert (result.pages, result.text_pages) == (4, 4)


def test_a_few_blank_pages_are_fine():
    assert check(make_pdf_bytes(TEN, blank={3})).text_pages == 9


def test_scanned_pdf_is_rejected():
    with pytest.raises(PdfRejected) as e:
        check(make_pdf_bytes(TEN, blank={1, 5}))  # 20% without text
    assert str(e.value) == SCANNED_MESSAGE


def test_watermarked_image_only_pages_count_as_scanned():
    with pytest.raises(PdfRejected, match="no text layer"):
        check(make_pdf_bytes(TEN, blank=set(range(10)), watermark=True))


def test_not_a_pdf():
    with pytest.raises(PdfRejected) as e:
        check(b"hello, this is not a pdf")
    assert str(e.value) == NOT_A_PDF_MESSAGE


def test_encrypted_pdf_is_rejected():
    doc = pymupdf.open(stream=make_pdf_bytes(BODIES), filetype="pdf")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(PdfRejected, match="couldn't be opened"):
        check(data)


def test_too_large():
    data = make_pdf_bytes(BODIES)
    with pytest.raises(PdfRejected, match="limit is"):
        check(data, max_bytes=len(data) - 1)


def test_too_many_pages():
    with pytest.raises(PdfRejected, match="4 pages; the limit is 3"):
        check(make_pdf_bytes(BODIES), max_pages=3)
