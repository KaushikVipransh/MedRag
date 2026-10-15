"""Multi-user MedRAG: accounts, a private library per user, and chat with one ready book."""
import logging

import streamlit as st

from medrag import services as services_module
from medrag.accounts import MIN_PASSWORD, AuthError
from medrag.cloud.base import BackendError, BackendUnavailable
from medrag.config import CloudSettings, Settings
from medrag.keys import KeyRejected
from medrag.library import UploadRejected
from medrag.safety import DISCLAIMER
from medrag.web.components import render_message_body

UNAVAILABLE_MESSAGE = "The storage service is waking up or unavailable. Try again in a minute."
ERROR_MESSAGE = "Something went wrong. Please try again in a moment."
PAGES = ("Chat", "My library", "Settings")
log = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading models… the first start takes about a minute.")
def get_services():
    return services_module.build_services(Settings.from_env(), CloudSettings.from_env())


def run() -> None:
    st.title("⚕️ MedRAG")
    try:
        s = get_services()
    except Exception as exc:
        st.error(f"MedRAG could not start: {exc}")
        st.stop()
    try:
        user = st.session_state.get("user")
        if user is None:
            login_page(s)
            return
        with st.sidebar:
            st.caption(f"Signed in as {user['email']}")
            page = st.radio("Go to", PAGES, key="page")
            if st.button("Log out", key="logout"):
                st.session_state.clear()
                st.rerun()
        {"Chat": chat_page, "My library": library_page, "Settings": settings_page}[page](s, user)
    except BackendUnavailable:
        st.warning(UNAVAILABLE_MESSAGE)
    except BackendError:
        log.exception("storage error")
        st.error(ERROR_MESSAGE)


def _enter(action) -> None:
    try:
        user = action()
    except AuthError as exc:
        st.error(str(exc))
        return
    st.session_state.user = {"id": user.id, "email": user.email}
    st.rerun()


def login_page(s) -> None:
    st.caption("Upload your own medical textbook and ask questions answered only from it, with page citations.")
    st.info(DISCLAIMER)
    log_in, sign_up = st.tabs(["Log in", "Sign up"])
    with log_in, st.form("login_form"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Log in", key="login_submit"):
            _enter(lambda: s.accounts.sign_in(email, password))
    with sign_up, st.form("signup_form"):
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        st.caption(f"At least {MIN_PASSWORD} characters. There is no password reset by email yet, so keep it safe.")
        if st.form_submit_button("Create account", key="signup_submit"):
            _enter(lambda: s.accounts.sign_up(new_email, new_password))


def _usage_line(s, user_id: str) -> str:
    cap = s.cloud.shared_daily_cap
    used = min(s.quota.used_today(user_id), cap)
    line = f"Free questions today: {used}/{cap}."
    if s.vault.has_key(user_id):
        line += " After that, your own Groq key is used."
    elif used >= cap:
        line += " Add your free Groq key in Settings to keep going."
    return line


def chat_page(s, user) -> None:
    uid = user["id"]
    ready = {b.id: b.title for b in s.library.list(uid) if b.status == "ready"}
    if not ready:
        st.info("You have no ready books yet. Upload one under **My library**.")
        return
    book_id = st.selectbox("Book", list(ready), format_func=ready.get, key="chat_book")
    st.info(DISCLAIMER)
    messages = st.session_state.setdefault("chats", {}).setdefault(book_id, [])
    for msg in messages:
        with st.chat_message(msg["role"]):
            render_message_body(msg)
    if messages and st.button("Clear chat"):
        messages.clear()
        st.rerun()
    st.caption(_usage_line(s, uid))
    if prompt := st.chat_input("Ask a question about this book…"):
        history = [{"role": m["role"], "content": m["content"]} for m in messages]
        messages.append({"role": "user", "content": prompt})
        with st.spinner("Searching the book…"):
            try:
                answer = s.chat.ask(uid, book_id, prompt, history)
            except BackendUnavailable:
                messages.pop()
                raise
            except Exception:
                log.exception("answering failed")
                messages.pop()
                st.error("Something went wrong while answering. Please try again.")
                return
        messages.append({"role": "assistant", "content": answer.text, "answer": answer, "question": prompt})
        st.rerun()


def library_page(s, user) -> None:
    uid = user["id"]
    c = s.cloud
    st.header("My library")
    st.caption(
        f"Upload a medical or health textbook as a PDF with selectable text; scanned PDFs can't be used. "
        f"Up to {c.max_books_per_user} books, each up to {c.max_pdf_mb} MB and {c.max_pages} pages."
    )
    with st.form("upload_form", clear_on_submit=True):
        file = st.file_uploader("PDF", type=["pdf"])
        title = st.text_input("Title (optional)")
        confirmed = st.checkbox("This is a medical or health book and I have the right to use it.")
        submitted = st.form_submit_button("Upload")
    if submitted:
        if file is None:
            st.error("Choose a PDF first.")
        else:
            try:
                with st.spinner("Checking the PDF…"):
                    book = services_module.upload_book(s, uid, title, file.name, file.getvalue(), confirmed)
                st.success(f"“{book.title}” was uploaded. Indexing has started and can take a few minutes.")
            except UploadRejected as exc:
                st.error(str(exc))
    books = s.library.list(uid)
    if any(b.status == "processing" for b in books):
        _live_books(s, uid)
    else:
        _render_books(s, uid, books)


@st.fragment(run_every=5)
def _live_books(s, uid: str) -> None:
    books = s.library.list(uid)
    _render_books(s, uid, books)
    if not any(b.status == "processing" for b in books):
        st.rerun()


def _render_books(s, uid: str, books) -> None:
    if not books:
        st.info("No books yet.")
        return
    for b in books:
        with st.container(border=True):
            st.markdown(f"**{b.title}**")
            if b.status == "ready":
                st.caption(f"Ready · {b.pages} pages · {b.chunks} passages · {b.filename}")
            elif b.status == "processing":
                st.caption(f"Indexing… {b.progress}")
            else:
                st.error(f"Indexing failed: {b.error}")
            if st.button("Delete", key=f"delete_{b.id}"):
                services_module.delete_book(s, uid, b.id)
                st.rerun()


def settings_page(s, user) -> None:
    uid = user["id"]
    cap = s.cloud.shared_daily_cap
    st.header("Settings")
    st.subheader("Your Groq API key")
    st.markdown(
        f"Everyone gets {cap} free questions a day on MedRAG's shared key. After that, MedRAG uses your own "
        "free Groq key: create one at [console.groq.com/keys](https://console.groq.com/keys). "
        "It is stored encrypted and never shown again in full."
    )
    with st.form("key_form", clear_on_submit=True):
        key = st.text_input("Groq API key", type="password", key="groq_key_input")
        if st.form_submit_button("Save key", key="save_key"):
            try:
                s.vault.save(uid, key)
                st.success("Key saved.")
            except KeyRejected as exc:
                st.error(str(exc))
    masked = s.vault.masked(uid)
    if masked:
        st.markdown(f"Saved key: `{masked}`")
        if st.button("Remove key", key="remove_key"):
            s.vault.remove(uid)
            st.rerun()
    else:
        st.caption("No key saved.")
    st.subheader("Today's usage")
    st.markdown(f"{min(s.quota.used_today(uid), cap)} of {cap} free questions used today (resets at 00:00 UTC).")
