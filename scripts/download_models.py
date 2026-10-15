"""Downloads the embedder, re-ranker and gate into the Hugging Face cache at image build time,
so a cold start doesn't download them."""
from medrag.config import Settings
from medrag.embeddings import SentenceTransformerEmbedder
from medrag.retrieval.rerank import CrossEncoderReranker

s = Settings()
SentenceTransformerEmbedder(s.embed_model)
CrossEncoderReranker(s.rerank_model)
CrossEncoderReranker(s.gate_model)
print("models cached:", s.embed_model, s.rerank_model, s.gate_model)
