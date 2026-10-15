"""Users' own Groq keys: checked with Groq once, stored Fernet-encrypted, shown only masked."""
import re
from typing import Callable

from cryptography.fernet import Fernet, InvalidToken

from medrag.cloud.base import Database

_KEY_FORMAT = re.compile(r"^gsk_[A-Za-z0-9]{20,}$")


class KeyRejected(ValueError):
    """str(error) is safe to show to the user."""


def mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}"


def check_groq_key(key: str) -> bool:
    """Lists models (free, no tokens). False if Groq rejects the key."""
    from groq import AuthenticationError, Groq

    try:
        Groq(api_key=key, max_retries=0, timeout=15.0).models.list()
    except AuthenticationError:
        return False
    except Exception as exc:
        raise KeyRejected("Couldn't reach Groq to check the key. Try again in a minute.") from exc
    return True


class KeyVault:
    def __init__(self, db: Database, fernet_key: str, checker: Callable[[str], bool] = check_groq_key):
        self._db = db
        self._fernet = Fernet(fernet_key.encode())
        self._check = checker

    def save(self, user_id: str, key: str) -> None:
        key = key.strip()
        if not _KEY_FORMAT.match(key):
            raise KeyRejected("That doesn't look like a Groq API key. It should start with gsk_.")
        if not self._check(key):
            raise KeyRejected("Groq rejected this key. Check it and try again.")
        self._db.set_user_key(user_id, self._fernet.encrypt(key.encode()).decode())

    def get(self, user_id: str) -> str | None:
        token = self._db.get_user_key(user_id)
        if not token:
            return None
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except InvalidToken:
            return None

    def has_key(self, user_id: str) -> bool:
        return self.get(user_id) is not None

    def masked(self, user_id: str) -> str | None:
        key = self.get(user_id)
        return mask(key) if key else None

    def remove(self, user_id: str) -> None:
        self._db.delete_user_key(user_id)
