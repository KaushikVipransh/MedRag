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


from medrag.generation.llm import LLMAuthError, LLMRateLimited


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


def _client_raising(exc):
    client = GroqClient(api_key="gsk_test", model="m")

    class Completions:
        def create(self, **kwargs):
            raise exc

    client._client = type("C", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    return client


def test_rate_limit_is_typed():
    with pytest.raises(LLMRateLimited, match="RateLimitError"):
        _client_raising(RateLimitError("daily tokens")).complete([{"role": "user", "content": "hi"}])


def test_bad_key_is_typed():
    with pytest.raises(LLMAuthError):
        _client_raising(AuthenticationError("invalid api key")).complete([{"role": "user", "content": "hi"}])


def test_other_errors_stay_generic():
    with pytest.raises(LLMError) as info:
        _client_raising(ValueError("boom")).complete([{"role": "user", "content": "hi"}])
    assert type(info.value) is LLMError
