from medrag.ingest.clean import (
    find_repeating_margin_lines,
    is_non_content_page,
    lines_to_text,
    strip_margin_lines,
)

HEADER = "Diseases of Ear, Nose and Throat"


def _pages():
    return [[HEADER, f"Body line {chr(65 + i)} about the ear.", str(i + 10)] for i in range(5)]


def test_running_header_and_page_number_are_detected():
    rep = find_repeating_margin_lines(_pages())
    assert HEADER.lower() in rep
    assert "#" in rep
    assert not any("body line" in r for r in rep)


def test_rare_margin_lines_are_not_treated_as_headers():
    pages = [["Rare heading", "body"], ["Rare heading", "body2"], ["other", "body3"]]
    assert "rare heading" not in find_repeating_margin_lines(pages, min_count=4)


def test_strip_margin_lines_keeps_body():
    pages = _pages()
    rep = find_repeating_margin_lines(pages)
    assert strip_margin_lines(pages[0], rep) == ["Body line A about the ear."]


def test_repeated_line_outside_margin_is_kept():
    lines = ["H1", "H2", "H3", HEADER, "x", "y", "z", "10"]
    rep = {HEADER.lower(), "#"}
    assert strip_margin_lines(lines, rep) == ["H1", "H2", "H3", HEADER, "x", "y", "z"]


def test_hyphenated_word_is_rejoined():
    assert lines_to_text(["The tympa-", "nic membrane is thin."]) == "The tympanic membrane is thin."


def test_hyphen_before_capital_is_kept():
    assert lines_to_text(["Bing-", "Siebenmann dysplasia"]) == "Bing-Siebenmann dysplasia"


def test_wrapped_lines_join_and_bullets_break():
    text = lines_to_text(["Causes include:", "• otosclerosis", "• Ménière's", "disease"])
    assert text == "Causes include:\n• otosclerosis\n• Ménière's disease"


def test_numbered_items_break():
    assert lines_to_text(["Types:", "(i) acute", "(ii) chronic"]) == "Types:\n(i) acute\n(ii) chronic"


def test_blank_line_breaks_paragraph():
    assert lines_to_text(["Para one.", "", "Para two."]) == "Para one.\nPara two."


def test_ligatures_are_expanded():
    assert lines_to_text(["signiﬁcant improvement"]) == "significant improvement"


def test_unassigned_char_after_bullet_is_treated_as_space():
    text = lines_to_text(["Tumours:", "•฀Lymphoma", "•฀Sarcoma"])
    assert text == "Tumours:\n• Lymphoma\n• Sarcoma"


def test_index_page_is_non_content():
    assert is_non_content_page([f"Term {i}, {100 + i}" for i in range(30)])


def test_contents_heading_is_non_content():
    assert is_non_content_page(["Contents", "Chapter 1 Anatomy of Ear 3"])


def test_page_mentioning_index_in_prose_is_content():
    assert not is_non_content_page(["Index finger approach to the ear canal is described below."] * 5)


def test_empty_page_is_non_content():
    assert is_non_content_page([" ", ""])


def test_recurring_subheading_that_also_appears_in_body_is_not_a_header():
    margin_pages = [[HEADER, f"body {i}", "more", "Treatment"] for i in range(5)]
    body_pages = [["a", "b", "c", "Treatment", "x", "y", "z"] for _ in range(3)]
    rep = find_repeating_margin_lines(margin_pages + body_pages)
    assert "treatment" not in rep
    assert HEADER.lower() in rep


def test_bare_page_numbers_are_headers_even_if_numbers_appear_in_tables():
    pages = [["Title line", f"text {i}", "x", "y", "12", "34", "56", "a", "b", str(i + 1)] for i in range(6)]
    assert "#" in find_repeating_margin_lines(pages)
