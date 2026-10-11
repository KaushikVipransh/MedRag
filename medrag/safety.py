import re

DISCLAIMER = (
    "MedRAG is an educational tool that answers from the indexed textbook(s). "
    "It is not medical advice and does not replace a qualified clinician."
)
EMERGENCY_MESSAGE = (
    "This sounds like it could be an emergency. Contact your local emergency number or go to "
    "the nearest emergency department now. Do not wait for an answer from this tool."
)

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_EMERGENCY = re.compile(
    r"\b(?:i|we|he|she|they|my \w+)\s+(?:can'?t|cannot|can not)\s+breathe"
    r"|\bnot breathing\b|\bchoking\b"
    r"|\bswallowed (?:a |an )?(?:button )?(?:battery|magnet)"
    r"|\b(?:won'?t|will not|doesn'?t|does not) stop bleeding"
    r"|\bsevere(?:ly)? bleeding\b|\bunconscious\b"
    r"|\bsuicid|\bkill myself\b",
    re.IGNORECASE,
)


def validate_question(question: str, max_chars: int) -> tuple[str | None, str | None]:
    clean = " ".join(_CONTROL.sub(" ", question).split())
    if not clean:
        return None, "Please enter a question."
    if len(clean) > max_chars:
        return None, f"Please keep questions under {max_chars} characters."
    return clean, None


def is_emergency(question: str) -> bool:
    return bool(_EMERGENCY.search(question))
