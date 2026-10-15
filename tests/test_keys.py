import pytest
from cryptography.fernet import Fernet

from medrag.cloud.memory import MemoryDatabase
from medrag.keys import KeyRejected, KeyVault, mask

GOOD = "gsk_" + "A1b2C3d4" * 6


@pytest.fixture
def db():
    return MemoryDatabase()


def vault(db, checker=lambda key: True, secret=None):
    return KeyVault(db, secret or Fernet.generate_key().decode(), checker=checker)


def test_save_encrypts_and_get_decrypts(db):
    v = vault(db)
    v.save("alice", f"  {GOOD}\n")
    assert v.get("alice") == GOOD
    assert GOOD not in db.get_user_key("alice")
    assert v.has_key("alice") and not v.has_key("bob")


def test_masked_shows_only_prefix_and_last_four(db):
    v = vault(db)
    v.save("alice", GOOD)
    assert v.masked("alice") == mask(GOOD) == f"gsk_…{GOOD[-4:]}"
    assert v.masked("bob") is None


def test_rejects_malformed_key_without_calling_groq(db):
    calls = []
    v = vault(db, checker=lambda k: calls.append(k) or True)
    with pytest.raises(KeyRejected, match="gsk_"):
        v.save("alice", "sk-not-groq")
    assert calls == [] and not v.has_key("alice")


def test_rejects_key_groq_refuses(db):
    with pytest.raises(KeyRejected, match="rejected"):
        vault(db, checker=lambda k: False).save("alice", GOOD)


def test_remove(db):
    v = vault(db)
    v.save("alice", GOOD)
    v.remove("alice")
    assert v.get("alice") is None


def test_wrong_secret_reads_as_no_key(db):
    vault(db).save("alice", GOOD)
    assert vault(db).get("alice") is None  # a different Fernet secret cannot decrypt
