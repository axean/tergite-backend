"""Tests for the settings and configs"""
import pytest

from app.services.kernel.utils.config import KernelConfig
from app.tests.conftest import FASTAPI_CLIENTS
from app.tests.utils.env import TEST_MSS_APP_TOKEN
from app.tests.utils.fixtures import get_fixture_path, load_json_fixture

_HARDWARE_CONFIG_JSON = load_json_fixture("hardware-config.json")
_YAML_HARDWARE_CONFIG_PATH = get_fixture_path("hardware-config.yml")


def test_load_hardware_yaml_config():
    """KernelConfig can load YAML into hardware config JSON"""
    conf = KernelConfig.from_yaml(_YAML_HARDWARE_CONFIG_PATH)
    expected = _HARDWARE_CONFIG_JSON
    got = conf.to_quantify()
    assert got == expected


@pytest.mark.parametrize("client", FASTAPI_CLIENTS)
def test_authenticated_mss_client(client):
    """The MSS client used to make requests to MSS is authenticated"""
    from app.utils.http import get_mss_client

    mss_client = get_mss_client()
    authorization_header = mss_client.headers.get("Authorization")
    assert authorization_header == f"Bearer {TEST_MSS_APP_TOKEN}"
