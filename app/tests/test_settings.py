"""Tests for the settings file"""
import pytest

from app.tests.conftest import FASTAPI_CLIENTS
from app.tests.utils.env import TEST_MSS_APP_TOKEN


@pytest.mark.parametrize("client", FASTAPI_CLIENTS)
def test_authenticated_mss_client(client):
    """The MSS client used to make requests to MSS is authenticated"""
    from app.utils.http import get_mss_client

    mss_client = get_mss_client()
    authorization_header = mss_client.headers.get("Authorization")
    assert authorization_header == f"Bearer {TEST_MSS_APP_TOKEN}"
