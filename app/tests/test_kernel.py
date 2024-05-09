"""Tests for the kernel service"""
from app.services.kernel.utils.config import KernelConfig
from app.tests.utils.fixtures import get_fixture_path, load_json_fixture

_HARDWARE_CONFIG_JSON = load_json_fixture("hardware-config.json")
_YAML_HARDWARE_CONFIG_PATH = get_fixture_path("hardware-config.yml")


def test_load_hardware_yaml_config():
    """KernelConfig can load YAML into hardware config JSON"""
    conf = KernelConfig.from_yaml(_YAML_HARDWARE_CONFIG_PATH)
    expected = _HARDWARE_CONFIG_JSON
    got = conf.to_quantify()
    assert got == expected
