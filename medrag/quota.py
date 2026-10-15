"""Which Groq key answers a question: the shared app key while the user is under today's cap
(UTC day), then the user's own key. Only answers the model actually generated count."""
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable

from medrag.cloud.base import Database
from medrag.keys import KeyVault
from medrag.types import Answer


@dataclass(frozen=True)
class KeyChoice:
    source: str  # "shared" | "own" | "none"
    api_key: str


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def counts_toward_cap(answer: Answer) -> bool:
    return not answer.refused or answer.refusal_reason == "not_in_context"


class Quota:
    def __init__(
        self, db: Database, vault: KeyVault, shared_key: str, daily_cap: int,
        today: Callable[[], date] = utc_today,
    ):
        self._db = db
        self._vault = vault
        self.shared_key = shared_key
        self.daily_cap = daily_cap
        self._today = today

    def used_today(self, user_id: str) -> int:
        return self._db.usage(user_id, self._today())

    def own_key(self, user_id: str) -> str | None:
        return self._vault.get(user_id)

    def choose(self, user_id: str) -> KeyChoice:
        if self.shared_key and self.used_today(user_id) < self.daily_cap:
            return KeyChoice("shared", self.shared_key)
        own = self.own_key(user_id)
        return KeyChoice("own", own) if own else KeyChoice("none", "")

    def record(self, user_id: str) -> int:
        return self._db.increment_usage(user_id, self._today())
