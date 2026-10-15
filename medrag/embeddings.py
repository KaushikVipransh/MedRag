import threading
from typing import Protocol


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...

    def count_tokens(self, text: str) -> int: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        # One model serves the ingest thread and chat requests; HF fast tokenizers are not
        # thread-safe ("Already borrowed"), so calls are serialised.
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            vectors = self._model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 256,
            )
        return vectors.tolist()

    def count_tokens(self, text: str) -> int:
        with self._lock:
            return len(self._model.tokenizer.tokenize(text))
