from medrag.safety import is_emergency, validate_question


def test_blank_question_rejected():
    assert validate_question("   \n", 1000) == (None, "Please enter a question.")


def test_too_long_question_rejected():
    clean, error = validate_question("a" * 1001, 1000)
    assert clean is None and "1000" in error


def test_question_is_normalised():
    assert validate_question("  what  is\ttinnitus?\x00 ", 1000) == ("what is tinnitus?", None)


def test_emergency_phrases():
    assert is_emergency("My child swallowed a button battery")
    assert is_emergency("I can't breathe and my throat is closing")
    assert is_emergency("my nose won't stop bleeding")


def test_educational_questions_are_not_emergencies():
    assert not is_emergency("What is tinnitus?")
    assert not is_emergency("What are the causes of epistaxis?")
