import pytest

from app.scripts.pi_pulse_amplitude import calibrate_pi_pulse_amplitude
from app.scripts.train_discriminator import train_discriminator

from .utils.fixtures import get_fixture_path, load_fixture

_EXECUTOR_CONFIG_YAML = get_fixture_path("simulator-backend.yml")
_DISCRIMINATOR_CONFIG = get_fixture_path("discriminator-config.yml")
_DISCRIMINATOR_CONFIG_MOCK = load_fixture("discriminator-config-mock.yml", fmt="yaml")


@pytest.mark.skip
def test_discriminator_script():
    discriminator_config = train_discriminator(
        _EXECUTOR_CONFIG_YAML, _DISCRIMINATOR_CONFIG
    )
    assert _DISCRIMINATOR_CONFIG_MOCK == discriminator_config


def test_pi_pulse_amplitude():
    amplitude = calibrate_pi_pulse_amplitude()
    assert amplitude == 0.08953062777292731
