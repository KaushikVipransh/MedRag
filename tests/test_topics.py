from medrag.ingest.topics import generic_headings, topic_headings
from medrag.types import context_text
from tests.fakes import make_chunk


def test_generic_headings_are_those_repeated_across_pages():
    pages = [["TREATMENT", "body"], ["TREATMENT", "x"], ["TREATMENT", "y"], ["WEGENER’S GRANULOMATOSIS", "z"]]
    assert generic_headings(pages, min_pages=3) == {"TREATMENT"}


def test_topic_headings_skip_generic_tables_and_spaced_letters():
    lines = ["Some prose.", "WEGENER’S GRANULOMATOSIS", "AETIOLOGY", "TABLE 33.1 DIFFERENCES", "S E C T I O N", "more prose"]
    assert topic_headings(lines, generic={"AETIOLOGY"}) == ["Wegener’s Granulomatosis"]


def test_wrapped_heading_lines_are_joined():
    assert topic_headings(["GRANULOMAS OF UNSPECIFIED ", "ORIGIN", "text"], generic=set()) == ["Granulomas Of Unspecified Origin"]


def test_short_or_numeric_caps_are_not_headings():
    assert topic_headings(["CT", "T1 N0", "HIV", "Body text"], generic=set()) == []


def test_context_text_prefixes_chapter_and_topic():
    c = make_chunk(0, "Biopsy shows vasculitis.", section="Section II > Chapter 28 - Granulomatous Diseases of Nose")
    c = c.__class__(**{**c.__dict__, "topic": "Wegener’s Granulomatosis"})
    assert context_text(c) == "Chapter 28 - Granulomatous Diseases of Nose — Wegener’s Granulomatosis: Biopsy shows vasculitis."
    assert context_text(make_chunk(1, "x", section="")) == "x"


def test_numbered_list_headings_and_fragments_are_not_topics():
    lines = ["3. ESTHESIONEUROBLASTOMA (SYN. OLFACTORY NEU-", "A. TUBOTYMPANIC TYPE", "(AJCC, 2002)", "(a) FURUNCLE"]
    assert topic_headings(lines, generic=set()) == []
