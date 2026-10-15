import threading
import time

import numpy as np


class _Tracker:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def __enter__(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.01)

    def __exit__(self, *exc):
        with self.lock:
            self.active -= 1


def test_tokenizer_and_encoder_are_never_used_concurrently(monkeypatch):
    """HF fast tokenizers are not safe to share across threads (ingest thread vs chat requests)."""
    import sentence_transformers

    from medrag.embeddings import SentenceTransformerEmbedder

    tracker = _Tracker()

    class FakeTokenizer:
        def tokenize(self, text):
            with tracker:
                return text.split()

    class FakeModel:
        def __init__(self, name, device=None):
            self.tokenizer = FakeTokenizer()

        def encode(self, texts, **kwargs):
            with tracker:
                return np.zeros((len(texts), 4))

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    embedder = SentenceTransformerEmbedder("fake")

    def work(n):
        for _ in range(5):
            embedder.encode(["a b"]) if n % 2 else embedder.count_tokens("a b c")

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker.peak == 1


def test_cross_encoder_is_never_used_concurrently(monkeypatch):
    import torch
    import transformers

    from medrag.retrieval.rerank import CrossEncoderReranker

    tracker = _Tracker()

    class FakeTokenizer:
        def __call__(self, queries, texts, **kwargs):
            with tracker:
                return {"input_ids": torch.zeros((len(texts), 2), dtype=torch.long)}

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, input_ids):
            with tracker:
                return type("Out", (), {"logits": torch.zeros((input_ids.shape[0], 1))})()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda name: FakeTokenizer())
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", lambda name: FakeModel())
    reranker = CrossEncoderReranker("fake")

    def work():
        for _ in range(5):
            reranker.score("q", ["a", "b"])

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker.peak == 1
