# This code is part of Tergite
#
# (C) Copyright Martin Ahindura 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests specific to the quantum executor service"""
import socket

import pytest

from ...libs.quantum_executor.quantify.executor import QuantifyExecutor
from ..utils.fixtures import get_fixture_path

_REAL_HARDWARE_EXECUTOR_CONFIG_FILE = get_fixture_path("real-executor-config.yml")


def test_attempts_to_connect_to_real_hardware():
    """Loads the config for the real hardware in the appropriate way"""
    QuantifyExecutor.close()

    with pytest.raises(socket.timeout):
        QuantifyExecutor(config_file=_REAL_HARDWARE_EXECUTOR_CONFIG_FILE)
