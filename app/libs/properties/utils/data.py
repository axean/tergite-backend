# This code is part of Tergite
#
# (C) Copyright Abdullah-Al Amin 2023
# (C) Copyright Martin Ahindura 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Dict, List, Literal, Optional, Union

from ..dtos import CalibrationValue
from .storage import get_component_property, set_component_property


def get_inner_value(
    calib_value: Union[int, CalibrationValue, None]
) -> Union[int, float]:
    """Extracts value from a possible calibration value

    Args:
        calib_value: the calibration value

    Returns:
        the internal value in the calibration value
    """
    if isinstance(calib_value, CalibrationValue):
        return calib_value.value
    return calib_value


def read_qubit_calibration_data(
    qubit_ids: List[str], qubit_params: List[str]
) -> List[Dict[str, Union[int, CalibrationValue, None]]]:
    """Reads the calibration of the qubits of the device from the store (redis)

    Args:
        qubit_ids: the unique identifiers of the qubits
        qubit_params: the parameters stored for each qubit

    Returns:
        a list of dictionaries of qubit parameters and their values
    """
    return [
        {
            param: int(qubit_id.strip("q"))
            if param == "id"
            else _read_calibration_value(
                component_type="qubit",
                component_id=qubit_id.strip("q"),
                prop_name=param,
            )
            for param in qubit_params
        }
        for qubit_id in qubit_ids
    ]


def read_resonator_calibration_data(
    qubit_ids: List[str], resonator_params: List[str]
) -> List[Dict[str, Union[int, CalibrationValue, None]]]:
    """Reads the calibration of the resonators of the device from the store (redis)

    Args:
        qubit_ids: the unique identifiers of the qubits
        resonator_params: the parameters stored for each resonator

    Returns:
        a list of dictionaries of resonator parameters and their values
    """
    return [
        {
            param: int(qubit_id.strip("q"))
            if param == "id"
            else _read_calibration_value(
                component_type="readout_resonator",
                component_id=qubit_id.strip("q"),
                prop_name=param,
            )
            for param in resonator_params
        }
        for qubit_id in qubit_ids
    ]


def read_coupler_calibration_data(
    coupler_ids: List[str], coupler_params: List[str]
) -> List[Dict[str, Union[int, CalibrationValue, None]]]:
    """Reads the calibration of the couplers of the device from the store (redis)

    Args:
        coupler_ids: the unique identifiers of the couplers
        coupler_params: the parameters stored for each coupler

    Returns:
        a list of dictionaries of coupler parameters and their values
    """
    return [
        {
            param: int(coupler_id.strip("u"))
            if param == "id"
            else _read_calibration_value(
                component_type="coupler",
                component_id=coupler_id.strip("c"),
                prop_name=param,
            )
            for param in coupler_params
        }
        for coupler_id in coupler_ids
    ]


def read_discriminator_data(
    qubit_ids: List[str],
    params: List[str],
) -> Dict[str, Dict[str, Optional[CalibrationValue]]]:
    """Reads the discriminator data for the device from the store (redis)

    Args:
        qubit_ids: the unique identifiers of the qubits
        params: the parameters stored for the discriminator

    Returns:
        a dictionary of qubit id and its lda discriminators
    """
    return {
        qubit_id: {
            param: _read_calibration_value(
                component_type="discriminator",
                component_id=qubit_id.strip("q"),
                prop_name=param,
            )
            for param in params
        }
        for qubit_id in qubit_ids
    }


def _read_calibration_value(
    component_type: Literal["qubit", "readout_resonator", "discriminator"],
    component_id: str,
    prop_name: str,
) -> Optional[CalibrationValue]:
    """Reads the calibration value of the given prop name of the given component

    Args:
        component_id: the ID of the component
        component_type: the type of component e.g. qubit, readout_resonator etc
        prop_name: the name of the property to read

    Returns:
        the calibration value of the property of the given component
    """
    result = get_component_property(
        component=component_type, name=prop_name, component_id=component_id
    )
    if result is not None:
        return CalibrationValue(date=result[1], **result[0].model_dump())


def set_qubit_calibration_data(data: List[Dict[str, Optional[Dict]]]):
    """Sets the calibration of the qubits of the device in the store (redis)

    Args:
        data: the calibration data for all the qubits of a given device
    """
    # FIXME: Use this at the start of the simulator or whenever an automatic recalibration occurs
    #   so that it can be picked up when new calibration data is requested
    for qubit_conf in data:
        qubit_id = str(qubit_conf["id"]["value"]).strip("q")
        for k, v in qubit_conf.items():
            if isinstance(v, dict):
                set_component_property(
                    component="qubit", name=k, component_id=qubit_id, **v
                )


def set_resonator_calibration_data(data: List[Dict[str, Optional[Dict]]]):
    """Sets the calibration of the resonators of the device in the store (redis)

    Args:
        data: the calibration data for all the resonators of a given device
    """
    # FIXME: Use this at the start of the simulator or whenever an automatic recalibration occurs
    #   so that it can be picked up when new calibration data is requested
    for resonator_conf in data:
        qubit_id = str(resonator_conf["id"]["value"]).strip("q")
        for k, v in resonator_conf.items():
            if isinstance(v, dict):
                set_component_property(
                    component="readout_resonator", name=k, component_id=qubit_id, **v
                )


def set_discriminator_data(data: Dict[str, Dict[str, Optional[Dict]]]):
    """Sets the discriminator data of the device in the store (redis)

    Args:
        data: the discriminator data of a given device
    """
    # FIXME: Use this at the start of the simulator or whenever an automatic recalibration occurs
    #   so that it can be picked up when new calibration data is requested
    for key, discriminator_conf in data.items():
        qubit_id = str(key).strip("q")
        for k, v in discriminator_conf.items():
            if isinstance(v, dict):
                set_component_property(
                    component="discriminator", name=k, component_id=qubit_id, **v
                )


def set_coupler_calibration_data(data: List[Dict[str, Optional[Dict]]]):
    """Sets the calibration of the couplers of the device in the store (redis)

    Args:
        data: the calibration data for all the couplers of a given device
    """
    # FIXME: Use this at the start of the simulator or whenever an automatic recalibration occurs
    #   so that it can be picked up when new calibration data is requested
    for coupler_conf in data:
        coupler_id = str(coupler_conf["id"]["value"]).strip("c")
        for k, v in coupler_conf.items():
            if isinstance(v, dict):
                set_component_property(
                    component="coupler", name=k, component_id=coupler_id, **v
                )


def attach_units_many(
    data: List[Dict[str, Union[str, float]]], units_map: Dict[str, str]
) -> List[Dict[str, Dict[Literal["value", "unit"], Union[str, float]]]]:
    """Attaches units to the values in a list of dicts

    Args:
        data: the records to be transformed
        units_map: the map of property name to its unit

    Returns:
        the list of records with values of form {"value": ..., "unit": ...}
    """
    return [attach_units(item, units_map) for item in data]


def attach_units(
    data: Dict[str, Union[str, float]], units_map: Dict[str, str]
) -> Dict[str, Dict[Literal["value", "unit"], Union[str, float]]]:
    """Attaches units to the values in the dict

    Args:
        data: the record to be transformed
        units_map: the map of property name to its unit

    Returns:
        the record with values of form {"value": ..., "unit": ...}
    """
    return {k: {"value": v, "unit": units_map.get(k, "")} for k, v in data.items()}
