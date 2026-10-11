import pytest

pytestmark = pytest.mark.slow


def test_sentence_transformer_embedder():
    from medrag.embeddings import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    a, b, c = emb.encode(["ringing in the ears", "tinnitus", "fracture of the femur"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))
    assert len(a) == 384
    assert dot(a, b) > dot(a, c)
    assert emb.count_tokens("tinnitus retraining therapy") > 0
