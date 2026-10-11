from medrag.ingest.chunking import chunk_text


def WORDS(s: str) -> int:
    return len(s.split())


SIX_WORD_SENTENCES = " ".join(f"Sentence number {i} has six words." for i in range(40))


def test_short_text_is_single_chunk():
    text = "Tinnitus is ringing in the ear."
    assert chunk_text(text, WORDS, max_tokens=50, overlap_tokens=10) == [text]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", WORDS, 50, 10) == []


def test_chunks_respect_max_tokens():
    chunks = chunk_text(SIX_WORD_SENTENCES, WORDS, max_tokens=20, overlap_tokens=6)
    assert len(chunks) > 1
    assert all(WORDS(c) <= 20 for c in chunks)


def test_consecutive_chunks_overlap_by_whole_sentences():
    chunks = chunk_text(SIX_WORD_SENTENCES, WORDS, max_tokens=20, overlap_tokens=6)
    assert chunks[0].endswith("Sentence number 2 has six words.")
    assert chunks[1].startswith("Sentence number 2 has six words.")


def test_overlong_sentence_is_split_on_words():
    chunks = chunk_text(" ".join(["word"] * 45), WORDS, max_tokens=20, overlap_tokens=5)
    assert [WORDS(c) for c in chunks] == [20, 20, 5]


def test_overlap_dropped_when_it_would_overflow():
    text = "Short one here. " + " ".join(["long"] * 19) + " end."
    chunks = chunk_text(text, WORDS, max_tokens=20, overlap_tokens=5)
    assert chunks == ["Short one here.", " ".join(["long"] * 19) + " end."]


def test_newlines_are_unit_boundaries():
    chunks = chunk_text("Causes:\n• wax\n• otitis", WORDS, max_tokens=2, overlap_tokens=0)
    assert chunks == ["Causes:", "• wax", "• otitis"]
