from datetime import date

import pytest
from cryptography.fernet import Fernet

from medrag.cloud.memory import MemoryDatabase
from medrag.keys import KeyVault
from medrag.quota import KeyChoice, Quota, counts_toward_cap
from medrag.types import Answer

OWN = "gsk_" + "Z9" * 12


class Clock:
    def __init__(self):
        self.day = date(2026, 10, 15)

    def __call__(self):
        return self.day


@pytest.fixture
def parts():
    db = MemoryDatabase()
    vault = KeyVault(db, Fernet.generate_key().decode(), checker=lambda k: True)
    clock = Clock()
    return db, vault, clock, Quota(db, vault, "gsk_shared", daily_cap=2, today=clock)


def test_shared_key_until_cap_then_none(parts):
    _, _, _, q = parts
    assert q.choose("alice") == KeyChoice("shared", "gsk_shared")
    q.record("alice")
    q.record("alice")
    assert q.used_today("alice") == 2
    assert q.choose("alice") == KeyChoice("none", "")


def test_own_key_after_cap(parts):
    _, vault, _, q = parts
    vault.save("alice", OWN)
    q.record("alice")
    q.record("alice")
    assert q.choose("alice") == KeyChoice("own", OWN)
    assert q.own_key("alice") == OWN


def test_usage_is_per_user(parts):
    _, _, _, q = parts
    q.record("alice")
    q.record("alice")
    assert q.choose("bob").source == "shared"
    assert q.used_today("bob") == 0


def test_cap_resets_at_utc_midnight(parts):
    _, _, clock, q = parts
    q.record("alice")
    q.record("alice")
    clock.day = date(2026, 10, 16)
    assert q.used_today("alice") == 0
    assert q.choose("alice").source == "shared"


def test_no_shared_key_configured_uses_own_key_or_none():
    db = MemoryDatabase()
    vault = KeyVault(db, Fernet.generate_key().decode(), checker=lambda k: True)
    q = Quota(db, vault, "", daily_cap=10)
    assert q.choose("alice").source == "none"
    vault.save("alice", OWN)
    assert q.choose("alice") == KeyChoice("own", OWN)


@pytest.mark.parametrize("refused, reason, counts", [
    (False, None, True),
    (True, "not_in_context", True),
    (True, "low_relevance", False),
    (True, "invalid_input", False),
    (True, "llm_error", False),
    (True, "rate_limited", False),
    (True, "key_rejected", False),
])
def test_only_generated_answers_count(refused, reason, counts):
    assert counts_toward_cap(Answer(text="x", refused=refused, refusal_reason=reason)) is counts
