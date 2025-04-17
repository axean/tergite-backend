# This code is part of Tergite
#
# (C) Copyright Abdullah-Al Amin 2023
# (C) Copyright Martin Ahindura 2024
# (C) Copyright Adilet Tuleouv 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Modified:
#
# - Martin Ahindura, 2023
# - Stefan Hill, 2024
# - Adilet Tuleuov, 2025
#
# CAUTION: This updater is currently also used in the tergite-autocalibration-lite repository!
# Any change on this file should be done in both repositories until they are eventually merged!
from typing import Dict, List, Optional, Union

from requests import Session

import settings

from .dtos import (
    BackendConfig,
    CouplerProps,
    CouplersCalibration,
    DeviceCalibrationV2,
    DeviceProperties,
    DeviceV1,
    DeviceV2,
    QubitCalibration,
    QubitProps,
    ReadoutResonatorProps,
    ResonatorCalibration,
)
from .utils.data import (
    attach_units,
    attach_units_many,
    get_inner_value,
    read_coupler_calibration_data,
    read_discriminator_data,
    read_qubit_calibration_data,
    read_resonator_calibration_data,
    set_coupler_calibration_data,
    set_discriminator_data,
    set_qubit_calibration_data,
    set_resonator_calibration_data,
)

_BACKEND_CONFIG: Optional[BackendConfig] = None


def get_backend_config() -> BackendConfig:
    """Returns the current system's backend configuration.

    Loads the static configuration from the backend_config.toml file and
    merges it with seed data (from either qiskit_pulse or quantify seed file)
    validated via Pydantic.
    """
    global _BACKEND_CONFIG
    if _BACKEND_CONFIG is None:
        # Load static configuration from the main backend_config.toml
        _BACKEND_CONFIG = BackendConfig.from_toml(
            settings.BACKEND_SETTINGS, seed_file=settings.CALIBRATION_SEED
        )
    return _BACKEND_CONFIG


def initialize_backend(
    backend_config: BackendConfig,
    mss_client: Session,
    mss_url: str,
    qubit_config: Optional[List[Dict[str, Union[float, str]]]] = None,
    resonator_config: Optional[List[Dict[str, Union[float, str]]]] = None,
    coupler_config: Optional[List[Dict[str, Union[float, str]]]] = None,
    discriminator_config: Optional[
        Dict[str, Dict[str, Dict[str, Union[float, str]]]]
    ] = None,
    is_standalone: bool = settings.IS_STANDALONE,
):
    """Runs a number of operations to initialize the backend

    Args:
        backend_config: the configuration of the backend
        mss_client: the requests Session to make requests to MSS with
        mss_url: the URL to MSS
        qubit_config: the qubit calibration data to initialize the backend with;
                defaults to what is in the calibration.seed.toml
        resonator_config: the resonator calibration data to initialize the backend with;
                defaults to what is in the calibration.seed.toml
        discriminator_config: the discriminator calibration data to initialize the backend with;
                defaults to what is in the calibration.seed.toml
        coupler_config: the coupler calibration data to initialize the backend with;
                defaults to what is in the calibration.seed.toml
        is_standalone: whether this backend is standalone or is connected to an MSS

    Raises:
        ValueError: error message from MSS when it attempts to update mss
    """
    # if it is a simulator, set its simulated calibration values in redis
    if backend_config.calibration_config is not None:
        calibration_config = backend_config.calibration_config

        # set qubit calibration data
        qubit_units = calibration_config.units.get("qubit", {})
        qubit_data = calibration_config.qubit if qubit_config is None else qubit_config
        qubit_data = attach_units_many(qubit_data, qubit_units)
        set_qubit_calibration_data(qubit_data)

        # set readout_resonator calibration data
        resonator_units = calibration_config.units.get("readout_resonator", {})
        resonator_data = resonator_config
        if resonator_config is None:
            resonator_data = calibration_config.readout_resonator
        resonator_data = attach_units_many(resonator_data, resonator_units)
        set_resonator_calibration_data(resonator_data)

        # set discriminator data
        disc_units = calibration_config.units.get("discriminators", {})
        discriminator_data = discriminator_config
        if discriminator_config is None:
            discriminator_data = calibration_config.discriminators

        for disc_conf in discriminator_data.values():
            disc_data = {
                qbit: attach_units(v, disc_units) for qbit, v in disc_conf.items()
            }
            set_discriminator_data(disc_data)

        # set coupler calibration data
        coupler_units = calibration_config.units.get("coupler", {})
        coupler_data = coupler_config
        if coupler_config is None:
            coupler_data = calibration_config.coupler  # Fetch from simulator_config

        coupler_data = attach_units_many(coupler_data, coupler_units)
        set_coupler_calibration_data(coupler_data)

    if not is_standalone:
        # update MSS of this backend's configuration
        send_backend_info_to_mss(
            backend_config=backend_config,
            mss_client=mss_client,
            mss_url=mss_url,
        )


def get_device_v2_info(
    backend_config: Optional[BackendConfig] = None,
) -> DeviceV2:
    """Retrieves this device's info in DeviceV2 format

    Args:
        backend_config: the BackendConfig instance for this device

    Returns:
        the deviceV2 info of the device
    """
    if backend_config is None:
        backend_config = get_backend_config()
    qubit_ids = backend_config.device_config.qubit_ids
    return DeviceV2(
        **backend_config.general_config.model_dump(),
        meas_map=backend_config.device_config.meas_map,
        qubit_ids=qubit_ids,
        gates=backend_config.gates,
        number_of_qubits=backend_config.general_config.num_qubits,
        is_online=backend_config.general_config.is_active,
        basis_gates=list(backend_config.gates.keys()),
        coupling_map=backend_config.device_config.coupling_map,
        coordinates=backend_config.device_config.coordinates,
        is_simulator=backend_config.general_config.simulator,
        coupling_dict=backend_config.device_config.coupling_dict,
        qubit_ids_coupler_map=backend_config.device_config.qubit_ids_coupler_map,
    )


def get_device_calibration_v2_info(
    backend_config: Optional[BackendConfig] = None,
) -> DeviceCalibrationV2:
    """Retrieves this device's calibration info in DeviceCalibrationV2 format

    Args:
        backend_config: the BackendConfig instance for this device

    Returns:
        the DeviceCalibrationV2 info of the device
    """
    if backend_config is None:
        backend_config = get_backend_config()

    qubit_ids = backend_config.device_config.qubit_ids
    discriminator_params = backend_config.device_config.discriminator_parameters
    discriminators = backend_config.device_config.discriminators

    # Couplng dicts is a dictionary of id as a keys and qubits list for each couplers
    coupler_ids = backend_config.device_config.coupling_dict.keys()

    raw_qubit_conf = read_qubit_calibration_data(
        qubit_ids=qubit_ids,
        qubit_params=backend_config.device_config.qubit_parameters,
    )
    qubit_conf = [QubitCalibration(**conf) for conf in raw_qubit_conf]
    last_calibrated: Optional[str] = None
    if len(qubit_conf) > 0:
        last_calibrated = qubit_conf[0].t2_decoherence.date

    raw_resonator_conf = read_resonator_calibration_data(
        qubit_ids=qubit_ids,
        resonator_params=backend_config.device_config.resonator_parameters,
    )
    resonator_conf = [ResonatorCalibration(**conf) for conf in raw_resonator_conf]
    raw_coupler_conf = read_coupler_calibration_data(
        coupler_ids=coupler_ids,
        coupler_params=backend_config.device_config.coupler_parameters,
    )
    coupler_conf = [CouplersCalibration(**conf) for conf in raw_coupler_conf]
    raw_discriminator_conf = {
        item: read_discriminator_data(
            qubit_ids=qubit_ids, params=discriminator_params[item]
        )
        for item in discriminators
    }
    discriminator_conf = {
        discriminator: {
            qubit_id: {k: get_inner_value(v) for k, v in conf.items()}
            for qubit_id, conf in discriminator_conf.items()
        }
        for discriminator, discriminator_conf in raw_discriminator_conf.items()
    }

    return DeviceCalibrationV2(
        name=backend_config.general_config.name,
        version=backend_config.general_config.version,
        qubits=qubit_conf,
        resonators=resonator_conf,
        couplers=coupler_conf,
        discriminators=discriminator_conf,
        last_calibrated=last_calibrated,
    )


def send_backend_info_to_mss(
    mss_client: Session,
    mss_url: str = settings.MSS_MACHINE_ROOT_URL,
    collection: str = None,
    backend_config: Optional[BackendConfig] = None,
):
    """
    Sends this backend's information to MSS

    Args:
        mss_client: the requests Session to run the queries
        mss_url: the URL to MSS
        collection: Please specify if backend should not be pushed to the standard collection in the DB
        backend_config: the BackendConfig instance for this device

    Raises:
        ValueError: error message from MSS
    """
    if backend_config is None:
        backend_config = get_backend_config()

    device_v2_info = get_device_v2_info(backend_config=backend_config).model_dump()
    calibration_v2_info = get_device_calibration_v2_info(
        backend_config=backend_config
    ).model_dump()

    responses = [
        mss_client.put(f"{mss_url}/v2/devices", json=device_v2_info),
        mss_client.post(f"{mss_url}/v2/calibrations", json=[calibration_v2_info]),
    ]

    error_message = ",".join([v.text for v in responses if not v.ok])
    if error_message != "":
        raise ValueError(error_message)
