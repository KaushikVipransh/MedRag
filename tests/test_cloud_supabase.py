import httpx
import pytest

from medrag.cloud.base import AuthFailed, BackendError, BackendUnavailable
from medrag.cloud.supabase import auth_failure, guard


class FakeAuthApiError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize("exc, code", [
    (FakeAuthApiError("x", code="user_already_exists"), "exists"),
    (FakeAuthApiError("x", code="email_exists"), "exists"),
    (FakeAuthApiError("A user with this email address has already been registered"), "exists"),
    (FakeAuthApiError("x", code="invalid_credentials"), "invalid_credentials"),
    (FakeAuthApiError("Invalid login credentials"), "invalid_credentials"),
    (FakeAuthApiError("x", code="weak_password"), "weak_password"),
    (FakeAuthApiError("Password should be at least 6 characters"), "weak_password"),
    (FakeAuthApiError("Database error saving new user"), "other"),
])
def test_auth_failure_codes(exc, code):
    failure = auth_failure(exc)
    assert isinstance(failure, AuthFailed) and failure.code == code


def test_guard_maps_network_errors_to_unavailable():
    @guard
    def call():
        raise httpx.ConnectError("connection refused")

    with pytest.raises(BackendUnavailable):
        call()


def test_guard_maps_server_errors_to_unavailable():
    response = httpx.Response(503, request=httpx.Request("GET", "https://x"))

    @guard
    def call():
        raise httpx.HTTPStatusError("503", request=response.request, response=response)

    with pytest.raises(BackendUnavailable):
        call()


def test_guard_maps_other_errors_to_backend_error_and_keeps_ours():
    @guard
    def broken():
        raise RuntimeError("violates foreign key constraint")

    with pytest.raises(BackendError):
        broken()

    @guard
    def ours():
        raise BackendUnavailable("paused")

    with pytest.raises(BackendUnavailable):
        ours()


def test_build_services_reports_missing_settings():
    from medrag.config import CloudSettings, Settings
    from medrag.services import build_services

    with pytest.raises(ValueError, match="SUPABASE_ANON_KEY"):
        build_services(Settings(), CloudSettings(backend="supabase", supabase_url="https://x"))
