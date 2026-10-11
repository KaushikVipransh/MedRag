from medrag.generation.llm import LLMClient, LLMError

CONDENSE_SYSTEM = (
    "Rewrite the user's latest message as one standalone question that can be understood "
    "without the conversation. Keep medical terms exactly as written. Do not answer it. "
    "Output only the rewritten question."
)


def condense_question(llm: LLMClient, question: str, history: list[dict], max_turns: int = 3) -> str:
    if not history:
        return question
    recent = history[-2 * max_turns :]
    transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent)
    messages = [
        {"role": "system", "content": CONDENSE_SYSTEM},
        {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"},
    ]
    try:
        rewritten = llm.complete(messages, temperature=0.0, max_tokens=400).strip()
    except LLMError:
        return question
    return rewritten if 0 < len(rewritten) <= 500 else question
