from medrag.ingest.sections import sections_from_fonts, sections_from_toc


def test_toc_assigns_sections_to_pages():
    toc = [[1, "Ear", 1], [2, "Anatomy", 1], [2, "Tinnitus", 3], [1, "Nose", 5]]
    assert sections_from_toc(toc, 6) == [
        "Ear > Anatomy", "Ear > Anatomy", "Ear > Tinnitus", "Ear > Tinnitus", "Nose", "Nose",
    ]


def test_toc_ignores_deep_levels_and_missing_pages():
    toc = [[1, "Ear", 1], [3, "Deep", 2], [2, "Lost", -1]]
    assert sections_from_toc(toc, 2) == ["Ear", "Ear"]


def test_pages_before_first_entry_are_blank():
    assert sections_from_toc([[1, "Ear", 2]], 3) == ["", "Ear", "Ear"]


def test_toc_titles_have_whitespace_normalised():
    assert sections_from_toc([[1, " Physiology of\xa0Auditory  System ", 1]], 1) == ["Physiology of Auditory System"]


def test_font_headings_carry_forward():
    pages = [
        [("Chapter 1 Ear", 16.0), ("body text " * 20, 10.0)],
        [("more body " * 20, 10.0)],
        [("Tinnitus", 16.0), ("body", 10.0)],
    ]
    assert sections_from_fonts(pages) == ["Chapter 1 Ear", "Chapter 1 Ear", "Tinnitus"]


def test_uniform_font_gives_no_sections():
    assert sections_from_fonts([[("a b c", 10.0)]]) == [""]


def test_no_text_gives_blank_sections():
    assert sections_from_fonts([[], []]) == ["", ""]
