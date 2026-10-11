from medrag.conversation import condense_question
from medrag.generation.llm import LLMError
from tests.fakes import FakeLLM

HISTORY = [
    {"role": "user", "content": "What is Meniere's disease?"},
    {"role": "assistant", "content": "It is endolymphatic hydrops [1]."},
]


def test_first_question_is_unchanged_and_llm_not_called():
    llm = FakeLLM()
    assert condense_question(llm, "What is tinnitus?", []) == "What is tinnitus?"
    assert llm.calls == []


def test_follow_up_is_rewritten():
    llm = FakeLLM(["  How is Meniere's disease treated?\n"])
    assert condense_question(llm, "how is it treated?", HISTORY) == "How is Meniere's disease treated?"
    prompt = llm.calls[0][-1]["content"]
    assert "Meniere's disease" in prompt and "how is it treated?" in prompt


def test_llm_failure_falls_back_to_original():
    llm = FakeLLM(error=LLMError("down"))
    assert condense_question(llm, "how is it treated?", HISTORY) == "how is it treated?"


def test_unusable_rewrite_falls_back_to_original():
    assert condense_question(FakeLLM([""]), "and in children?", HISTORY) == "and in children?"
    assert condense_question(FakeLLM(["x" * 600]), "and in children?", HISTORY) == "and in children?"
