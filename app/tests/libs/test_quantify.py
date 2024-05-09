"""Tests for the quantify lib"""
from app.libs.quantify.utils.config import QuantifyConfig
from app.tests.utils.fixtures import get_fixture_path, load_json_fixture

_HARDWARE_CONFIG_JSON = load_json_fixture("hardware-config.json")
_YAML_HARDWARE_CONFIG_PATH = get_fixture_path("hardware-config.yml")


def test_load_hardware_yaml_config():
    """QuantifyConfig can load YAML into hardware config JSON"""
    conf = QuantifyConfig.from_yaml(_YAML_HARDWARE_CONFIG_PATH)
    expected = _HARDWARE_CONFIG_JSON
    got = conf.to_quantify()
    assert got == expected
