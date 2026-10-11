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


def test_cross_encoder_prefers_relevant_passage():
    from medrag.retrieval.rerank import CrossEncoderReranker

    rr = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    rel, irrel = rr.score(
        "What is tinnitus?",
        ["Tinnitus is a ringing sound in the ear originating within the patient.",
         "The femur is the longest bone in the human body."],
    )
    assert 0.0 <= irrel < rel <= 1.0
