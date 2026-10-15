"""Sign-up and log-in. A user is identified by the id the auth backend issues; the app keeps
that id in the Streamlit session and passes it to every data call."""
import re
from dataclasses import dataclass

from medrag.cloud.base import Auth, AuthFailed

MIN_PASSWORD = 8
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MESSAGES = {
    "exists": "An account with this email already exists. Log in instead.",
    "invalid_credentials": "Wrong email or password.",
    "weak_password": f"Choose a stronger password (at least {MIN_PASSWORD} characters).",
    "other": "Sign-in failed. Please try again in a moment.",
}


class AuthError(ValueError):
    """A sign-up or log-in problem; str(error) is safe to show to the user."""


@dataclass(frozen=True)
class User:
    id: str
    email: str


def _clean_email(email: str) -> str:
    cleaned = email.strip().lower()
    if not _EMAIL.match(cleaned):
        raise AuthError("Enter a valid email address.")
    return cleaned


class Accounts:
    def __init__(self, auth: Auth):
        self.auth = auth

    def sign_up(self, email: str, password: str) -> User:
        email = _clean_email(email)
        if len(password) < MIN_PASSWORD:
            raise AuthError(_MESSAGES["weak_password"])
        try:
            return User(self.auth.create_user(email, password), email)
        except AuthFailed as exc:
            raise AuthError(_MESSAGES.get(exc.code, _MESSAGES["other"])) from exc

    def sign_in(self, email: str, password: str) -> User:
        email = _clean_email(email)
        if not password:
            raise AuthError("Enter your password.")
        try:
            return User(self.auth.verify_password(email, password), email)
        except AuthFailed as exc:
            raise AuthError(_MESSAGES.get(exc.code, _MESSAGES["other"])) from exc
