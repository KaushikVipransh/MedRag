import pytest

from medrag.accounts import MIN_PASSWORD, Accounts, AuthError
from medrag.cloud.base import AuthFailed, BackendUnavailable
from medrag.cloud.memory import MemoryAuth


@pytest.fixture
def accounts():
    return Accounts(MemoryAuth())


def test_sign_up_then_sign_in(accounts):
    user = accounts.sign_up("  Alice@Example.org ", "password1")
    assert user.email == "alice@example.org"
    assert accounts.sign_in("alice@example.org", "password1") == user


def test_duplicate_email(accounts):
    accounts.sign_up("a@x.org", "password1")
    with pytest.raises(AuthError, match="already exists"):
        accounts.sign_up("A@X.org", "password2")


def test_wrong_password(accounts):
    accounts.sign_up("a@x.org", "password1")
    with pytest.raises(AuthError, match="Wrong email or password"):
        accounts.sign_in("a@x.org", "nope-nope")


@pytest.mark.parametrize("email", ["", "no-at-sign", "a@b", "a b@x.org"])
def test_invalid_email(accounts, email):
    with pytest.raises(AuthError, match="valid email"):
        accounts.sign_up(email, "password1")


def test_short_password(accounts):
    with pytest.raises(AuthError, match=str(MIN_PASSWORD)):
        accounts.sign_up("a@x.org", "short")


def test_empty_password_on_sign_in(accounts):
    with pytest.raises(AuthError, match="password"):
        accounts.sign_in("a@x.org", "")


def test_unknown_backend_failure_is_generic():
    class Broken:
        def create_user(self, email, password):
            raise AuthFailed("other", "500 internal")

    with pytest.raises(AuthError, match="try again"):
        Accounts(Broken()).sign_up("a@x.org", "password1")


def test_unavailable_backend_propagates():
    class Down:
        def verify_password(self, email, password):
            raise BackendUnavailable("paused")

    with pytest.raises(BackendUnavailable):
        Accounts(Down()).sign_in("a@x.org", "password1")
