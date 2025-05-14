# This code is part of Tergite
#
# (C) Copyright Martin Ahindura 2024
# (C) Copyright Chalmers Next Labs 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simple utility functions for the execution worker"""
from typing import Optional

from redis import Redis

import settings
from app.libs.properties import get_backend_config, initialize_backend
from app.libs.quantum_executor.base.executor import QuantumExecutor
from app.libs.quantum_executor.qiskit.executor import QiskitDynamicsExecutor
from app.libs.quantum_executor.quantify.executor import QuantifyExecutor
from app.utils.api import get_mss_client


def get_executor(
    redis: Redis = settings.REDIS_CONNECTION,
    executor_type: str = settings.EXECUTOR_TYPE,
    quantify_config_file: str = settings.QUANTIFY_CONFIG_FILE,
    quantify_metadata_file: str = settings.QUANTIFY_METADATA_FILE,
    mss_url: str = settings.MSS_MACHINE_ROOT_URL,
) -> QuantumExecutor:
    """Gets the executor for running jobs

    It also initializes the backend before returning the executor

    Args:
        redis: the connection to the redis database
        executor_type: the executor type to return
        quantify_config_file: the path to the configuration file of the executor
        quantify_metadata_file: the path to the metadata file of the executor
        mss_url: the URL to MSS

    Returns:
        An initialized quantum executor
    """
    executor: Optional[QuantumExecutor] = None
    backend_config = get_backend_config()

    if executor_type == "quantify":
        executor = QuantifyExecutor(
            quantify_config_file=quantify_config_file,
            quantify_metadata_file=quantify_metadata_file,
            backend_config=backend_config,
        )

    if executor_type == "qiskit_pulse_1q":
        executor: QiskitDynamicsExecutor = QiskitDynamicsExecutor.new_one_qubit(
            backend_config=backend_config
        )
        backend_config.calibration_config.discriminators = (
            executor.backend.train_discriminator()
        )

    if executor_type == "qiskit_pulse_2q":
        executor: QiskitDynamicsExecutor = QiskitDynamicsExecutor.new_two_qubit(
            backend_config=backend_config
        )
        backend_config.calibration_config.discriminators = (
            executor.backend.train_discriminator()
        )

    initialize_backend(
        redis,
        mss_client=get_mss_client(),
        mss_url=mss_url,
        backend_config=backend_config,
    )

    return executor
