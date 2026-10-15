"""Round trip against a real Supabase project. Needs SUPABASE_URL, SUPABASE_ANON_KEY,
SUPABASE_SERVICE_KEY and MEDRAG_FERNET_KEY (from .env). Uses fake models and a fake LLM, so it
costs no Groq quota. Run: docker compose run --rm dev pytest -m slow tests/test_live_supabase.py"""
import uuid

import pytest

from medrag.config import CloudSettings, Settings
from medrag.services import Models, assemble, delete_book, make_backend, upload_book
from tests.fakes import FakeEmbedder, OverlapReranker
from tests.fakes_cloud import BODIES, KeyedLLM, make_pdf_bytes

pytestmark = pytest.mark.slow


@pytest.fixture
def live():
    cloud = CloudSettings.from_env()
    if cloud.backend != "supabase" or cloud.missing():
        pytest.skip("needs Supabase settings in .env")
    backend = make_backend(cloud)
    settings = Settings(groq_api_key="gsk_fake", chunk_max_tokens=50, chunk_overlap_tokens=10)
    models = Models(FakeEmbedder(), OverlapReranker(), OverlapReranker(), KeyedLLM())
    services = assemble(backend, models, settings, cloud, key_checker=lambda k: True, threaded=False)
    created: list[str] = []
    yield services, created
    for user_id in created:
        backend.auth.delete_user(user_id)


def test_sign_up_upload_ask_delete(live):
    services, created = live
    email = f"medrag-test-{uuid.uuid4().hex[:10]}@example.com"
    user = services.accounts.sign_up(email, "correct-horse-battery")
    created.append(user.id)
    assert services.accounts.sign_in(email, "correct-horse-battery").id == user.id

    book = upload_book(services, user.id, "Live test", "live.pdf", make_pdf_bytes(BODIES), confirmed=True)
    ready = services.library.get(user.id, book.id)
    assert ready.status == "ready" and ready.chunks > 0

    answer = services.chat.ask(user.id, book.id, "What is tinnitus?")
    assert not answer.refused and services.quota.used_today(user.id) == 1

    stranger = services.accounts.sign_up(f"medrag-test-{uuid.uuid4().hex[:10]}@example.com", "correct-horse-battery")
    created.append(stranger.id)
    assert services.library.get(stranger.id, book.id) is None
    assert services.chat.ask(stranger.id, book.id, "What is tinnitus?").refusal_reason == "book_unavailable"

    assert delete_book(services, user.id, book.id)
    assert services.library.get(user.id, book.id) is None
    assert services.backend.db.load_chunks(user.id, book.id) == []
