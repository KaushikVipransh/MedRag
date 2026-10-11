from typing import Protocol


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...

    def count_tokens(self, text: str) -> int: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return vectors.tolist()

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.tokenize(text))
