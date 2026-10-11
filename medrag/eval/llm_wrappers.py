import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable

from medrag.generation.llm import LLMClient, LLMError

class QuotaExhausted(RuntimeError):
    """Raised when the provider's daily quota is spent. Deliberately NOT an LLMError, so
    the pipeline and judges do not turn it into a recorded failure: the run aborts and
    resumes later from the cache."""


_WAIT = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE)


class CachedLLM:
    """Disk cache keyed by (model, messages, params, salt). Change `salt` to force fresh
    samples, e.g. for repeated consistency runs. Empty outputs are never cached."""

    def __init__(self, inner: LLMClient, model: str, cache_dir: Path, salt: str = ""):
        self.inner = inner
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.salt = salt
        self.hits = 0
        self.misses = 0

    def _path(self, messages, temperature, max_tokens) -> Path:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": temperature,
             "max_tokens": max_tokens, "salt": self.salt},
            sort_keys=True, ensure_ascii=False,
        )
        return self.cache_dir / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        path = self._path(messages, temperature, max_tokens)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        self.misses += 1
        text = self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens)
        if text.strip():
            path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
        return text


def wait_seconds(message: str, default: float = 20.0) -> float:
    m = _WAIT.search(message)
    if not m:
        return default
    hours, minutes, seconds = int(m.group(1) or 0), int(m.group(2) or 0), float(m.group(3))
    return hours * 3600 + minutes * 60 + seconds + 1.0


def _is_rate_limit(message: str) -> bool:
    m = message.lower()
    return "ratelimit" in m.replace(" ", "") or "rate limit" in m or "429" in m


class PatientLLM:
    """Retries rate-limited calls, sleeping as long as the provider asks. A wait longer
    than `max_wait` means the daily quota is spent: fail fast so the run can resume later."""

    def __init__(
        self,
        inner: LLMClient,
        max_attempts: int = 10,
        sleep: Callable[[float], None] = time.sleep,
        default_wait: float = 20.0,
        max_wait: float = 900.0,
    ):
        self.inner = inner
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.default_wait = default_wait
        self.max_wait = max_wait

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens)
            except LLMError as exc:
                message = str(exc)
                if not _is_rate_limit(message) or attempt == self.max_attempts:
                    raise
                wait = wait_seconds(message, self.default_wait)
                if wait > self.max_wait:
                    raise QuotaExhausted(
                        f"Groq daily quota exhausted (asked to wait {wait / 60:.0f} min). "
                        f"Re-run later; finished work is cached. Original: {message}"
                    ) from exc
                self.sleep(wait)
        raise AssertionError("unreachable")
