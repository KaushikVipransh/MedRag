from medrag.generation.citations import (
    is_refusal,
    parse_citations,
    remove_invalid_markers,
    uncited_sentences,
)
from medrag.generation.llm import LLMClient, LLMError
from medrag.generation.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_message
from medrag.types import Answer, Citation, RetrievedChunk

LLM_UNAVAILABLE_MESSAGE = "The language model is unavailable right now. Please try again in a moment."


class Generator:
    def __init__(self, llm: LLMClient, temperature: float = 0.1):
        self.llm = llm
        self.temperature = temperature

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, chunks)},
        ]
        try:
            raw = self.llm.complete(messages, temperature=self.temperature, max_tokens=2048)
        except LLMError as exc:
            return Answer(
                text=LLM_UNAVAILABLE_MESSAGE, refused=True, refusal_reason="llm_error",
                chunks=chunks, warnings=[str(exc)],
            )

        if not raw.strip():
            return Answer(
                text=LLM_UNAVAILABLE_MESSAGE, refused=True, refusal_reason="llm_error",
                chunks=chunks, warnings=["empty response from language model"],
            )
        if is_refusal(raw):
            return Answer(text=NOT_FOUND_MESSAGE, refused=True, refusal_reason="not_in_context", chunks=chunks)

        valid, invalid = parse_citations(raw, len(chunks))
        text = remove_invalid_markers(raw, len(chunks)).strip()
        warnings = []
        if invalid:
            warnings.append(f"removed citations to non-existent sources: {invalid}")
        uncited = uncited_sentences(text)
        if uncited:
            warnings.append(f"{len(uncited)} sentence(s) without a citation")
        if not valid:
            warnings.append("answer cites no sources")
        return Answer(
            text=text,
            refused=False,
            citations=[Citation(n, chunks[n - 1].chunk) for n in valid],
            chunks=chunks,
            warnings=warnings,
        )
