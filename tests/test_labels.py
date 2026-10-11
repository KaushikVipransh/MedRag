from medrag.ingest.labels import infer_page_labels


def test_offset_inferred_from_footer_numbers():
    pages = [["Cover"], ["Preface text"], ["Contents"]] + [["Body text", str(i - 2)] for i in range(3, 10)]
    labels = infer_page_labels(pages)
    assert labels[3] == "1" and labels[9] == "7"
    assert labels[0] == "front-1" and labels[2] == "front-3"


def test_numbers_inside_body_do_not_vote():
    pages = [
        ["Line a", "Line b", "Line c", f"Table {i + 50} shows", "Line d", "Line e", "Line f", str(i + 1)]
        for i in range(5)
    ]
    assert infer_page_labels(pages)[:2] == ["1", "2"]


def test_number_in_header_with_title():
    pages = [[f"{i + 20}  Diseases of Ear"] + ["body"] * 6 for i in range(4)]
    assert infer_page_labels(pages)[0] == "20"


def test_embedded_pdf_labels_win():
    assert infer_page_labels([["x"], ["y"]], pdf_labels=["iv", "1"]) == ["iv", "1"]


def test_trivial_pdf_labels_are_ignored():
    pages = [["Body", str(i + 5)] for i in range(3)]
    assert infer_page_labels(pages, pdf_labels=["1", "2", "3"]) == ["5", "6", "7"]


def test_no_numbers_falls_back_to_pdf_index():
    assert infer_page_labels([["a"], ["b"]]) == ["1", "2"]
