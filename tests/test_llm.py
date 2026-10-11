import pytest

from medrag.generation.llm import GroqClient, LLMError


def test_missing_key_raises_llm_error():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqClient(api_key="", model="openai/gpt-oss-120b")


@pytest.mark.slow
def test_groq_roundtrip():
    from medrag.config import Settings

    s = Settings.from_env()
    out = GroqClient(s.groq_api_key, s.llm_model).complete(
        [{"role": "user", "content": "Reply with the single word: ok"}], temperature=0.0, max_tokens=200
    )
    assert "ok" in out.lower()
