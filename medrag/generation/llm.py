from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str: ...


class GroqClient:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set. Add it to .env (see .env.example).")
        from groq import Groq

        self._client = Groq(api_key=api_key, max_retries=3, timeout=60.0)
        self.model = model

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as exc:  # network, auth, rate limit, model retired
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        return resp.choices[0].message.content or ""
