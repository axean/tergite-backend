"""Tests for the settings and configs"""
import os

import pytest
import requests

from app.libs.quantum_executor.utils.config import QuantifyExecutorConfig
from app.tests.conftest import FASTAPI_CLIENTS
from app.tests.utils.env import TEST_BACKEND_SETTINGS_FILE, TEST_MSS_APP_TOKEN
from app.tests.utils.fixtures import get_fixture_path, load_fixture
from app.tests.utils.modules import remove_modules

_QUANTIFY_CONFIG_JSON = load_fixture("generic-quantify-config.json")
_YAML_QUANTIFY_CONFIG_PATH = get_fixture_path("generic-quantify-config.yml")


def test_load_hardware_yaml_config():
    """ExecutorConfig can load YAML into hardware config JSON"""
    conf = QuantifyExecutorConfig.from_yaml(_YAML_QUANTIFY_CONFIG_PATH)
    expected = _QUANTIFY_CONFIG_JSON
    got = conf.to_quantify()
    assert got == expected


@pytest.mark.parametrize("client", FASTAPI_CLIENTS)
def test_authenticated_mss_client(client):
    """The MSS client used to make requests to MSS is authenticated"""
    from app.utils.http import get_mss_client

    mss_client = get_mss_client()
    authorization_header = mss_client.headers.get("Authorization")
    assert authorization_header == f"Bearer {TEST_MSS_APP_TOKEN}"


def test_is_standalone(async_standalone_backend_client):
    """Raises no connection errors when is standalone and MSS is unavailable"""
    with async_standalone_backend_client as client:
        response = client.get("/")
        assert response.status_code == 200


def test_redis_connection(real_redis_client):
    """The global redis client as be found in settings"""
    from settings import REDIS_CONNECTION

    # Write a value in the real redis connection
    expected = "123"
    REDIS_CONNECTION.set("abc", expected)
    # Read it from the test client
    got = real_redis_client.get("abc").decode()
    assert expected == got
    real_redis_client.flushall()


def test_no_mss_connected():
    """Raises connection errors only when MSS is unavailable and is not standalone"""
    remove_modules(["os", "app", "settings"])

    os.environ["EXECUTOR_TYPE"] = "quantify"
    os.environ["BACKEND_SETTINGS"] = TEST_BACKEND_SETTINGS_FILE
    os.environ["IS_STANDALONE"] = "False"

    with pytest.raises(requests.exceptions.ConnectionError):
        from app.api import app
