from medrag.retrieval.bm25 import BM25Index, tokenize
from tests.fakes import make_chunk

CHUNKS = [
    make_chunk(0, "Meniere disease causes vertigo, tinnitus and deafness."),
    make_chunk(1, "Sinusitis, acute or chronic, is treated medically."),
    make_chunk(2, "The nasal septum has three parts."),
    make_chunk(3, "Otosclerosis causes conductive hearing loss."),
    make_chunk(4, "Tonsillitis is common in children.", doc="book-b"),
]


def test_tokenize_folds_accents_case_and_punctuation():
    assert tokenize("Ménière's Disease, SINUSITIS!") == ["meniere", "s", "disease", "sinusitis"]


def test_tokenize_drops_stopwords():
    assert tokenize("What is the cause of it?") == ["cause"]


def test_accented_query_matches_plain_text():
    hits = BM25Index(CHUNKS).search("Ménière's disease", n=3)
    assert hits[0][0].id == CHUNKS[0].id


def test_punctuation_in_corpus_does_not_block_match():
    hits = BM25Index(CHUNKS).search("sinusitis treatment", n=3)
    assert hits[0][0].id == CHUNKS[1].id


def test_query_without_terms_returns_nothing():
    assert BM25Index(CHUNKS).search("what is the?", n=3) == []
    assert BM25Index(CHUNKS).search("???", n=3) == []


def test_doc_filter():
    hits = BM25Index(CHUNKS).search("tonsillitis children", n=3, doc_ids=["book-a"])
    assert hits == []


def test_empty_index():
    assert BM25Index([]).search("ear", n=3) == []
