from medrag.generation.citations import (
    is_refusal,
    parse_citations,
    remove_invalid_markers,
    uncited_sentences,
)


def test_parse_valid_and_invalid():
    assert parse_citations("A [2]. B [1][7]. C [2, 3].", n_sources=3) == ([2, 1, 3], [7])


def test_remove_invalid_markers():
    assert remove_invalid_markers("A [1][7]. B [2, 9]. C [8].", 3) == "A [1]. B [2]. C."


def test_uncited_sentence_detected():
    text = "Tinnitus is a ringing sound that originates inside the patient. It can be treated with maskers [1]."
    assert uncited_sentences(text) == ["Tinnitus is a ringing sound that originates inside the patient."]


def test_citation_after_full_stop_counts():
    text = "Tinnitus is a ringing sound that originates inside the patient. [1] Maskers are worn like hearing aids [2]."
    assert uncited_sentences(text) == []


def test_short_lines_and_lead_ins_are_ignored():
    assert uncited_sentences("Treatment:\n- Maskers [1]\nSee above.") == []


def test_refusal_detection():
    assert is_refusal("NOT_FOUND")
    assert is_refusal("  not_found\n")
    assert not is_refusal("Tinnitus is ringing [1].")


def test_alternative_citation_styles_are_normalised():
    from medrag.generation.citations import normalize_markers

    text = "A【1】 B【2†L1-L3】 C[3†L4] D [1, 2]."
    assert normalize_markers(text) == "A[1] B[2] C[3] D [1, 2]."
