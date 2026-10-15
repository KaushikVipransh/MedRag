"""Helpers for multi-user tests: small generated PDFs (more helpers are added in Task 11)."""
import pymupdf

BODIES = [
    "Tinnitus is ringing in the ear.\nIt may be subjective or objective.\nMaskers help some patients.\nRetraining aims at habituation.",
    "Otosclerosis causes conductive deafness.\nStapedectomy is the usual operation.\nHearing aids are an option.\nFluoride may slow it.",
    "The nasal septum has three parts.\nThese are columellar, membranous\nand septum proper.\nDeviation causes obstruction.",
    "Acute sinusitis follows a cold.\nMaxillary sinus is most affected.\nTreatment uses antibiotics.\nDecongestants aid drainage.",
]


def make_pdf_bytes(bodies: list[str], blank=(), watermark: bool = False) -> bytes:
    """One page per body with a running header and a printed page number from 101.
    Pages listed in `blank` get no body: only the header, number and optional watermark,
    which is what an image-only (scanned) page looks like to text extraction."""
    doc = pymupdf.open()
    for n, body in enumerate(bodies):
        page = doc.new_page()
        page.insert_text((72, 60), "Test Book of ENT")
        if n not in blank:
            page.insert_text((72, 120), body)
        page.insert_text((300, 800), str(n + 101))
        if watermark:
            page.insert_text((72, 820), "mebooksfree.com")
    data = doc.tobytes()
    doc.close()
    return data


from cryptography.fernet import Fernet

from medrag.cloud.memory import memory_backend
from medrag.config import CloudSettings, Settings
from tests.fakes import FakeEmbedder, OverlapReranker

REPLY = "Tinnitus is ringing in the ear [1]."


class KeyedLLM:
    """LLM factory for tests: every client replies `reply`, except keys listed in `errors`,
    whose clients raise that error. `keys_used` records which key served each call."""

    def __init__(self, reply: str = REPLY, errors: dict | None = None):
        self.reply = reply
        self.errors = dict(errors or {})
        self.keys_used: list[str] = []

    def __call__(self, api_key: str):
        factory = self

        class Client:
            def complete(self, messages, temperature=0.0, max_tokens=1024):
                factory.keys_used.append(api_key)
                if api_key in factory.errors:
                    raise factory.errors[api_key]
                return factory.reply

        return Client()


def make_services(shared_key: str = "gsk_shared", cap: int = 2, llm: KeyedLLM | None = None, **cloud_overrides):
    from medrag.services import Models, assemble

    llm = llm or KeyedLLM()
    settings = Settings(groq_api_key=shared_key, rerank_threshold=0.045, top_n=3,
                        chunk_max_tokens=50, chunk_overlap_tokens=10)
    cloud = CloudSettings(backend="memory", fernet_key=Fernet.generate_key().decode(),
                          shared_daily_cap=cap, **cloud_overrides)
    models = Models(embedder=FakeEmbedder(), reranker=OverlapReranker(), gate=OverlapReranker(), llm_factory=llm)
    services = assemble(memory_backend(), models, settings, cloud, key_checker=lambda k: True, threaded=False)
    return services, llm


def ready_book(services, user_id: str, bodies=None, title: str = "ENT Notes"):
    from medrag.services import upload_book

    book = upload_book(services, user_id, title, "ent.pdf", make_pdf_bytes(bodies or BODIES), confirmed=True)
    return services.library.get(user_id, book.id)
