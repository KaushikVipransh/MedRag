from medrag.config import Settings
from medrag.conversation import condense_question
from medrag.generation.generator import Generator
from medrag.generation.llm import LLMClient
from medrag.generation.prompts import NOT_FOUND_MESSAGE
from medrag.retrieval.retriever import HybridRetriever
from medrag.safety import is_emergency, validate_question
from medrag.types import Answer


class MedRAG:
    def __init__(self, retriever: HybridRetriever, generator: Generator, llm: LLMClient, settings: Settings):
        self.retriever = retriever
        self.generator = generator
        self.llm = llm
        self.settings = settings

    def answer(
        self, question: str, history: list[dict] | None = None, doc_ids: list[str] | None = None
    ) -> Answer:
        clean, error = validate_question(question, self.settings.max_question_chars)
        if error:
            return Answer(text=error, refused=True, refusal_reason="invalid_input")

        standalone = condense_question(self.llm, clean, history or [])
        chunks = self.retriever.retrieve(standalone, doc_ids)

        best = chunks[0].rerank_score if chunks else None
        if not chunks or (best is not None and best < self.settings.rerank_threshold):
            answer = Answer(text=NOT_FOUND_MESSAGE, refused=True, refusal_reason="low_relevance", chunks=chunks)
        else:
            answer = self.generator.generate(standalone, chunks)

        answer.standalone_question = standalone
        answer.emergency = is_emergency(clean)
        return answer


def build_pipeline(settings: Settings) -> MedRAG:
    from medrag.embeddings import SentenceTransformerEmbedder
    from medrag.generation.llm import GroqClient
    from medrag.retrieval.bm25 import BM25Index
    from medrag.retrieval.rerank import CrossEncoderReranker
    from medrag.store import ChunkStore

    store = ChunkStore(settings.chroma_dir, settings.collection)
    if store.count() == 0:
        raise RuntimeError(
            f"The index at {settings.chroma_dir} is empty. Run: python -m medrag.ingest data/sources"
        )
    llm = GroqClient(settings.groq_api_key, settings.llm_model)
    retriever = HybridRetriever(
        store=store,
        embedder=SentenceTransformerEmbedder(settings.embed_model),
        bm25=BM25Index(store.all_chunks()),
        reranker=CrossEncoderReranker(settings.rerank_model),
        candidates=settings.candidates_per_retriever,
        top_n=settings.top_n,
    )
    return MedRAG(retriever, Generator(llm, settings.temperature), llm, settings)
