import pytest

from medrag.eval.llm_wrappers import CachedLLM, PatientLLM, wait_seconds
from medrag.generation.llm import LLMError

MSGS = [{"role": "user", "content": "hi"}]


class Scripted:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, messages, temperature=0.0, max_tokens=1024):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_cache_hit_skips_inner(tmp_path):
    inner = Scripted("answer")
    llm = CachedLLM(inner, "m", tmp_path)
    assert llm.complete(MSGS) == "answer"
    assert llm.complete(MSGS) == "answer"
    assert inner.calls == 1 and llm.hits == 1 and llm.misses == 1


def test_cache_key_includes_params_and_salt(tmp_path):
    inner = Scripted("a", "b", "c")
    llm = CachedLLM(inner, "m", tmp_path)
    llm.complete(MSGS, temperature=0.0)
    llm.complete(MSGS, temperature=0.1)
    llm.salt = "rep1"
    llm.complete(MSGS, temperature=0.0)
    assert inner.calls == 3


def test_empty_output_is_not_cached(tmp_path):
    inner = Scripted("  ", "real")
    llm = CachedLLM(inner, "m", tmp_path)
    assert llm.complete(MSGS).strip() == ""
    assert llm.complete(MSGS) == "real"


def test_wait_seconds_parsing():
    assert wait_seconds("Please try again in 7.66s.") == pytest.approx(8.66)
    assert wait_seconds("try again in 1m2.5s") == pytest.approx(63.5)
    assert wait_seconds("try again in 1h2m3s") == pytest.approx(3724.0)
    assert wait_seconds("no hint here", default=20.0) == 20.0


def test_patient_retries_rate_limit_with_requested_wait():
    slept = []
    inner = Scripted(LLMError("RateLimitError: Error code: 429 - try again in 2.5s"), "ok")
    assert PatientLLM(inner, sleep=slept.append).complete(MSGS) == "ok"
    assert slept == [pytest.approx(3.5)]


def test_patient_does_not_retry_other_errors():
    inner = Scripted(LLMError("AuthenticationError: 401"))
    with pytest.raises(LLMError, match="401"):
        PatientLLM(inner, sleep=lambda s: None).complete(MSGS)
    assert inner.calls == 1


def test_patient_gives_up_after_max_attempts():
    inner = Scripted(*[LLMError("RateLimitError 429")] * 3)
    with pytest.raises(LLMError):
        PatientLLM(inner, max_attempts=3, sleep=lambda s: None).complete(MSGS)
    assert inner.calls == 3


def test_patient_gives_up_immediately_on_daily_quota():
    from medrag.eval.llm_wrappers import QuotaExhausted

    inner = Scripted(LLMError("RateLimitError 429 requests per day: try again in 3h10m2s"), "never")
    with pytest.raises(QuotaExhausted, match="daily") as info:
        PatientLLM(inner, sleep=lambda s: None).complete(MSGS)
    assert not isinstance(info.value, LLMError)  # must not be swallowed as an ordinary LLM failure
    assert inner.calls == 1
