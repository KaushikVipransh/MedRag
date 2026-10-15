import pymupdf
import pytest

from medrag.ingest.build import ingest_pdf, slugify
from medrag.store import ChunkStore
from tests.fakes import FakeEmbedder

EAR_1 = "Tinnitus is ringing in the ear.\nIt may be subjective or objective.\nMaskers help some patients.\nRetraining aims at habituation."
EAR_2 = "Otosclerosis causes conductive deafness.\nStapedectomy is the usual operation.\nHearing aids are an option.\nFluoride may slow it."
NOSE_1 = "The nasal septum has three parts.\nThese are columellar, membranous\nand septum proper.\nDeviation causes obstruction."
NOSE_2 = "Acute sinusitis follows a cold.\nMaxillary sinus is most affected.\nTreatment uses antibiotics.\nDecongestants aid drainage."


def make_pdf(path, bodies, toc=None, blank=()):
    doc = pymupdf.open()
    for n, body in enumerate(bodies):
        page = doc.new_page()
        if n in blank:
            continue
        page.insert_text((72, 60), "Test Book of ENT")
        page.insert_text((72, 120), body)
        page.insert_text((300, 800), str(n + 101))
    if toc:
        doc.set_toc(toc)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def store(tmp_path):
    return ChunkStore(tmp_path / "db", "test")


def test_slugify():
    assert slugify("Dhingra ENT 7e") == "dhingra-ent-7e"


def test_ingest_builds_citable_chunks(tmp_path, store):
    pdf = tmp_path / "ent-book.pdf"
    make_pdf(pdf, [EAR_1, EAR_2, NOSE_1, NOSE_2], toc=[[1, "Ear", 1], [1, "Nose", 3]])
    stats = ingest_pdf(pdf, store, FakeEmbedder(), max_tokens=50, overlap_tokens=10, title="ENT Test Book")

    assert stats.doc_id == "ent-book"
    assert stats.pages_total == 4 and stats.pages_indexed == 4
    assert stats.chunks == store.count() > 0
    chunks = store.all_chunks()
    first = min(chunks, key=lambda c: c.id)
    assert first.page_label == "101"
    assert first.section == "Ear"
    assert first.doc_title == "ENT Test Book"
    assert {c.section for c in chunks} == {"Ear", "Nose"}
    assert all("Test Book of ENT" not in c.text for c in chunks)
    assert any("Tinnitus is ringing in the ear." in c.text for c in chunks)


def test_reingest_replaces_instead_of_duplicating(tmp_path, store):
    pdf = tmp_path / "ent-book.pdf"
    make_pdf(pdf, [EAR_1, EAR_2, NOSE_1, NOSE_2])
    first = ingest_pdf(pdf, store, FakeEmbedder(), 50, 10)
    second = ingest_pdf(pdf, store, FakeEmbedder(), 50, 10)
    assert store.count() == first.chunks == second.chunks


def test_blank_pages_are_reported(tmp_path, store):
    pdf = tmp_path / "scan.pdf"
    make_pdf(pdf, [EAR_1, EAR_2, NOSE_1, ""], blank={3})
    stats = ingest_pdf(pdf, store, FakeEmbedder(), 50, 10)
    assert stats.pages_without_text == 1
    assert stats.pages_indexed == 3


def test_title_defaults_to_filename(tmp_path, store):
    pdf = tmp_path / "my_ent-book.pdf"
    make_pdf(pdf, [EAR_1, EAR_2, NOSE_1, NOSE_2])
    stats = ingest_pdf(pdf, store, FakeEmbedder(), 50, 10)
    assert stats.title == "My Ent Book"


def test_watermarked_image_only_page_is_reported(tmp_path, store):
    pdf = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    for n, body in enumerate([EAR_1, EAR_2, NOSE_1, NOSE_2, ""]):
        page = doc.new_page()
        page.insert_text((72, 60), "Test Book of ENT")
        if body:
            page.insert_text((72, 120), body)
        page.insert_text((72, 820), "mebooksfree.com")
    doc.save(str(pdf))
    doc.close()
    stats = ingest_pdf(pdf, store, FakeEmbedder(), 50, 10)
    assert stats.pages_without_text == 1


def test_chunks_carry_the_topic_heading_across_pages(tmp_path, store):
    pdf = tmp_path / "topics.pdf"
    bodies = [
        "TREATMENT\nTreatment text one is here and long enough.\nMore words for the page body.\nAnd more words.",
        "WEGENER'S GRANULOMATOSIS\nIt affects nose and kidneys.\nBiopsy shows granulomas here.\nMore body text.",
        "Biopsy shows necrotizing vasculitis.\nErythrocyte sedimentation rate is raised.\nTREATMENT\nSteroids and cyclophosphamide.",
        "TREATMENT\nTreatment text four is here and long enough.\nMore words for the page body.\nAnd more words.",
    ]
    make_pdf(pdf, bodies, toc=[[1, "Nose", 1]])
    ingest_pdf(pdf, store, FakeEmbedder(), 50, 10)
    third = [c for c in store.all_chunks() if c.page_index == 2]
    assert third and all(c.topic == "Wegener's Granulomatosis" for c in third)


from medrag.ingest.build import chunks_for_pdf
from tests.fakes_cloud import make_pdf_bytes


def test_chunks_for_pdf_reads_bytes_like_a_file(tmp_path):
    data = make_pdf_bytes([EAR_1, EAR_2, NOSE_1, NOSE_2])
    path = tmp_path / "book.pdf"
    path.write_bytes(data)
    from_bytes, stats = chunks_for_pdf(data, "b1", "Book", FakeEmbedder().count_tokens, 50, 10)
    from_path, _ = chunks_for_pdf(path, "b1", "Book", FakeEmbedder().count_tokens, 50, 10)
    assert from_bytes == from_path and stats.chunks == len(from_bytes) > 0
    assert stats.pages_total == 4 and from_bytes[0].doc_id == "b1"
