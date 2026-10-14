from medrag.generation.citations import is_refusal
from medrag.generation.generator import LLM_UNAVAILABLE_MESSAGE
from medrag.generation.llm import LLMClient, LLMError
from medrag.generation.prompts import NOT_FOUND_MESSAGE, REFUSAL_TOKEN
from medrag.types import Answer

NO_RAG_SYSTEM = (
    "You are a medical assistant. Answer the user's medical question concisely and accurately "
    f"from your own knowledge. If you do not know the answer, reply with exactly: {REFUSAL_TOKEN}"
)


class NoRagBaseline:
    """The same LLM with no retrieval: the comparison point for Objective 4."""

    def __init__(self, llm: LLMClient, temperature: float = 0.1):
        self.llm = llm
        self.temperature = temperature

    def answer(self, question: str, history=None, doc_ids=None) -> Answer:
        messages = [{"role": "system", "content": NO_RAG_SYSTEM}, {"role": "user", "content": question}]
        try:
            raw = self.llm.complete(messages, temperature=self.temperature, max_tokens=2048)
        except LLMError as exc:
            return Answer(LLM_UNAVAILABLE_MESSAGE, refused=True, refusal_reason="llm_error", warnings=[str(exc)])
        if not raw.strip():
            return Answer(LLM_UNAVAILABLE_MESSAGE, refused=True, refusal_reason="llm_error")
        if is_refusal(raw):
            return Answer(NOT_FOUND_MESSAGE, refused=True, refusal_reason="not_in_context")
        return Answer(text=raw.strip(), refused=False, standalone_question=question)
