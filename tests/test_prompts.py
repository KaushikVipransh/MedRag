from medrag.generation.prompts import REFUSAL_TOKEN, SYSTEM_PROMPT, build_user_message
from medrag.types import RetrievedChunk
from tests.fakes import make_chunk


def test_sources_are_numbered_with_page_and_section():
    chunks = [
        RetrievedChunk(make_chunk(0, "Tinnitus is ringing.", page=159, section="Ear > Tinnitus"), score=0.9),
        RetrievedChunk(make_chunk(1, "Maskers help.", page=160), score=0.8),
    ]
    msg = build_user_message("What is tinnitus?", chunks)
    assert "[1] (Book A, page 160, Ear > Tinnitus)\nTinnitus is ringing." in msg
    assert "[2] (Book A, page 161, Ear)\nMaskers help." in msg
    assert msg.rstrip().endswith("Question: What is tinnitus?")


def test_system_prompt_states_the_rules():
    assert REFUSAL_TOKEN in SYSTEM_PROMPT
    assert "[2]" in SYSTEM_PROMPT  # shows the citation format
