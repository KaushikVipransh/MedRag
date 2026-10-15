from medrag.chat import (
    BOOK_UNAVAILABLE_MESSAGE,
    OWN_KEY_LIMIT_MESSAGE,
    OWN_KEY_REJECTED_MESSAGE,
    QUOTA_MESSAGE,
    SHARED_EXHAUSTED_MESSAGE,
)
from medrag.generation.llm import LLMAuthError, LLMRateLimited
from medrag.services import delete_book
from tests.fakes_cloud import KeyedLLM, make_services, ready_book

OWN = "gsk_" + "Q7" * 12


def test_answers_from_the_book_on_the_shared_key_and_counts_it():
    s, llm = make_services()
    book = ready_book(s, "alice")
    answer = s.chat.ask("alice", book.id, "What is tinnitus?")
    assert not answer.refused and answer.citations
    assert answer.citations[0].chunk.doc_id == book.id and answer.citations[0].chunk.doc_title == "ENT Notes"
    assert llm.keys_used == ["gsk_shared"]
    assert s.quota.used_today("alice") == 1


def test_off_topic_question_is_refused_free():
    s, llm = make_services()
    book = ready_book(s, "alice")
    answer = s.chat.ask("alice", book.id, "Who won the football world cup?")
    assert answer.refusal_reason == "low_relevance"
    assert llm.keys_used == [] and s.quota.used_today("alice") == 0


def test_model_saying_not_found_still_counts():
    s, _ = make_services(llm=KeyedLLM(reply="NOT_FOUND"))
    book = ready_book(s, "alice")
    assert s.chat.ask("alice", book.id, "What is tinnitus?").refusal_reason == "not_in_context"
    assert s.quota.used_today("alice") == 1


def test_after_the_cap_without_own_key_the_user_is_told_to_add_one():
    s, llm = make_services(cap=1)
    book = ready_book(s, "alice")
    s.chat.ask("alice", book.id, "What is tinnitus?")
    answer = s.chat.ask("alice", book.id, "What is tinnitus?")
    assert answer.refusal_reason == "quota" and answer.text == QUOTA_MESSAGE.format(cap=1)
    assert llm.keys_used == ["gsk_shared"]


def test_after_the_cap_the_own_key_is_used_and_not_counted():
    s, llm = make_services(cap=1)
    s.vault.save("alice", OWN)
    book = ready_book(s, "alice")
    s.chat.ask("alice", book.id, "What is tinnitus?")
    assert not s.chat.ask("alice", book.id, "What is tinnitus?").refused
    assert llm.keys_used == ["gsk_shared", OWN]
    assert s.quota.used_today("alice") == 1


def test_shared_key_exhausted_falls_back_to_own_key():
    s, llm = make_services(llm=KeyedLLM(errors={"gsk_shared": LLMRateLimited("RateLimitError: tokens per day")}))
    s.vault.save("alice", OWN)
    book = ready_book(s, "alice")
    answer = s.chat.ask("alice", book.id, "What is tinnitus?")
    assert not answer.refused and llm.keys_used == ["gsk_shared", OWN]
    assert s.quota.used_today("alice") == 0


def test_shared_key_exhausted_without_own_key():
    s, _ = make_services(llm=KeyedLLM(errors={"gsk_shared": LLMRateLimited("RateLimitError")}))
    book = ready_book(s, "alice")
    answer = s.chat.ask("alice", book.id, "What is tinnitus?")
    assert answer.refused and answer.text == SHARED_EXHAUSTED_MESSAGE
    assert s.quota.used_today("alice") == 0


def test_own_key_problems_are_explained():
    s, _ = make_services(cap=0, llm=KeyedLLM(errors={OWN: LLMAuthError("AuthenticationError")}))
    s.vault.save("alice", OWN)
    book = ready_book(s, "alice")
    assert s.chat.ask("alice", book.id, "What is tinnitus?").text == OWN_KEY_REJECTED_MESSAGE
    s2, _ = make_services(cap=0, llm=KeyedLLM(errors={OWN: LLMRateLimited("RateLimitError")}))
    s2.vault.save("alice", OWN)
    book2 = ready_book(s2, "alice")
    assert s2.chat.ask("alice", book2.id, "What is tinnitus?").text == OWN_KEY_LIMIT_MESSAGE


def test_cannot_ask_about_someone_elses_book():
    s, llm = make_services()
    book = ready_book(s, "alice")
    answer = s.chat.ask("bob", book.id, "What is tinnitus?")
    assert answer.refusal_reason == "book_unavailable" and answer.text == BOOK_UNAVAILABLE_MESSAGE
    assert llm.keys_used == [] and s.quota.used_today("bob") == 0


def test_not_ready_or_deleted_books_are_unavailable():
    s, llm = make_services()
    from tests.fakes_cloud import BODIES, make_pdf_bytes

    pending = s.library.add("alice", "", "a.pdf", make_pdf_bytes(BODIES))
    assert s.chat.ask("alice", pending.id, "What is tinnitus?").refusal_reason == "book_unavailable"
    book = ready_book(s, "alice")
    delete_book(s, "alice", book.id)
    assert s.chat.ask("alice", book.id, "What is tinnitus?").refusal_reason == "book_unavailable"
    assert llm.keys_used == []


def test_retrievers_are_cached_per_book_with_lru_eviction():
    s, _ = make_services(cap=100, cached_books=2, max_books_per_user=3)
    loads = []
    original = s.backend.db.load_chunks
    s.chat.db.load_chunks = lambda u, b: (loads.append(b), original(u, b))[1]
    a, b, c = (ready_book(s, "alice", title=t) for t in ("A", "B", "C"))
    for book in (a, a, b, c, a):
        s.chat.ask("alice", book.id, "What is tinnitus?")
    assert loads == [a.id, b.id, c.id, a.id]
    assert s.chat.cached_ids() == [c.id, a.id]
