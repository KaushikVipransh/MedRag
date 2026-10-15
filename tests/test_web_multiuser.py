from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from medrag.cloud.base import BackendUnavailable
from tests.fakes_cloud import make_services, ready_book

APP = str(Path(__file__).resolve().parent.parent / "app.py")
OWN = "gsk_" + "K8" * 12


@pytest.fixture
def env(monkeypatch):
    import streamlit as st

    import medrag.services

    services, llm = make_services()
    monkeypatch.setenv("MEDRAG_BACKEND", "memory")
    monkeypatch.setattr(medrag.services, "build_services", lambda settings, cloud: services)
    st.cache_resource.clear()
    return services, llm


def app_as(services, user=None, page=None):
    at = AppTest.from_file(APP, default_timeout=30)
    if user is not None:
        at.session_state["user"] = {"id": user.id, "email": user.email}
    if page is not None:
        at.session_state["page"] = page
    return at.run()


def text_of(at):
    return " ".join(e.value for e in [*at.markdown, *at.caption, *at.info, *at.error, *at.success, *at.warning])


def test_sign_up_logs_you_in(env):
    services, _ = env
    at = app_as(services)
    at.text_input(key="signup_email").input("new@example.org")
    at.text_input(key="signup_password").input("password123")
    at.button(key="signup_submit").click().run()
    assert not at.exception
    assert at.session_state["user"]["email"] == "new@example.org"
    assert "Signed in as new@example.org" in text_of(at)


def test_wrong_password_shows_error(env):
    services, _ = env
    services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services)
    at.text_input(key="login_email").input("a@example.org")
    at.text_input(key="login_password").input("wrong-password")
    at.button(key="login_submit").click().run()
    assert "Wrong email or password." in text_of(at)
    assert "user" not in at.session_state


def test_chat_answers_with_sources(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    ready_book(services, user.id)
    at = app_as(services, user)
    at.chat_input[0].set_value("What is tinnitus?").run()
    assert not at.exception
    assert "Tinnitus is ringing in the ear [1]." in text_of(at)
    assert any("p. 101" in e.label for e in at.expander)
    assert "Free questions today: 1/2." in text_of(at)


def test_chat_without_books_points_to_library(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services, user)
    assert "no ready books" in text_of(at)


def test_library_lists_and_deletes(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    book = ready_book(services, user.id)
    at = app_as(services, user, page="My library")
    assert "ENT Notes" in text_of(at) and "Ready · 4 pages" in text_of(at)
    at.button(key=f"delete_{book.id}").click().run()
    assert services.library.list(user.id) == []
    assert "No books yet." in text_of(at)


def test_settings_saves_masks_and_removes_key(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services, user, page="Settings")
    at.text_input(key="groq_key_input").input(OWN)
    at.button(key="save_key").click().run()
    assert f"gsk_…{OWN[-4:]}" in text_of(at) and OWN not in text_of(at)
    at.button(key="remove_key").click().run()
    assert services.vault.get(user.id) is None


def test_bad_key_is_explained(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services, user, page="Settings")
    at.text_input(key="groq_key_input").input("not-a-key")
    at.button(key="save_key").click().run()
    assert "should start with gsk_" in text_of(at)


def test_log_out(env):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")
    at = app_as(services, user)
    at.button(key="logout").click().run()
    assert "user" not in at.session_state
    assert at.button(key="login_submit") is not None


def test_backend_outage_shows_banner(env, monkeypatch):
    services, _ = env
    user = services.accounts.sign_up("a@example.org", "password123")

    def down(user_id):
        raise BackendUnavailable("paused")

    monkeypatch.setattr(services.library, "list", down)
    at = app_as(services, user)
    assert not at.exception
    assert "waking up or unavailable" in text_of(at)
