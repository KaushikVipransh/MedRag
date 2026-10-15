import threading
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly. Sigmoid is applied explicitly so scores are
    0-1 regardless of library defaults, which keeps the refusal threshold stable."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self._lock = threading.Lock()  # shared by concurrent chat requests; fast tokenizers are not thread-safe

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        with self._lock, self._torch.no_grad():
            enc = self._tokenizer(
                [query] * len(texts), texts,
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            )
            logits = self._model(**enc).logits[:, 0]
            return self._torch.sigmoid(logits).tolist()
