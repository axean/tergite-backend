# This code is part of Tergite
#
# (C) Copyright David Wahlstedt 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import argparse
import logging
from typing import Optional

import toml

from backend_properties_config.initialize_properties import (
    initialize_properties,
    set_component_ids,
)
from backend_properties_storage.storage import (
    BackendProperty,
    PropertyType,
    set_component_property,
)

"""Logging initialization"""

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
# The following two lines are not used yet, but can be good to have available:
logger.setLevel(logging.INFO)
LOGLEVEL = logging.INFO


# =============================================================================
# Loading of the layout, property templates, and device configuration
#
# The layout properties will be saved in Redis as device properties as
# well, but int this configuration file we label them 'layout', to
# distinguish them from the other properties. The layout is loaded
# first, then the initialize_properties will use the
# property_templates to register which properties are supported, and
# what meta-information they have. After that, the device section is
# loaded, to populate declared properties with additional values/meta
# information.

# =============================================================================
# Configuration loading functions


def load_device_layout_configuration(layout: dict) -> bool:
    """Creates BackendProperty instances from the 'layout' section of
    the device configuration, and writes them into Redis.
    """

    property_type = PropertyType.DEVICE
    source = "configuration"

    try:
        component_tags = list(layout.keys())
        # Store which components are supported by the configuration
        BackendProperty(
            name="components",
            property_type=property_type,
            value=component_tags,
            source=source,
        ).write()
        for (component_tag, id_dict) in layout.items():
            # Store number_of_ for this kind of component.
            # id_dict contains the component ids from the layout configuration
            set_component_ids(component_tag, sorted(list(id_dict.keys())))
            # Store layout properties for each id
            for component_id, properties in id_dict.items():
                # Store properties for each id
                for property_name, property_value in properties.items():
                    set_component_property(
                        name=property_name,
                        component=component_tag,
                        component_id=component_id,
                        value=property_value,
                        source=source,
                    )
    except Exception as error:
        logger.error(f"Failed to load layout: {error=}")
        return False
    return True


def load_device_configuration(device_config: dict) -> bool:
    """Creates BackendProperty instances from the 'device' section of
    the device configuration, and writes them into Redis.
    """

    try:
        # get the component types defined by the device layout configuration file
        components = BackendProperty.read_value(
            property_type=PropertyType.DEVICE, name="components"
        )

        for tag, tag_value in device_config.items():
            if tag in components:
                # tag is a component type
                for component_id, properties in tag_value.items():
                    for name, value in properties.items():
                        if not _save_device_property(
                            name,
                            component=tag,
                            component_id=component_id,
                            value=value,
                        ):
                            return False
            elif isinstance(tag_value, dict):
                # A section whose name is the property
                # tag_value is the dict of property fields
                if not _save_device_property(
                    name=tag,
                    **tag_value,
                ):
                    return False
            else:
                # A single binding with just 'key', 'value'
                # tag is the property name
                if not _save_device_property(
                    name=tag,
                    value=tag_value,
                ):
                    return False
    except Exception as error:
        logger.error(f"Failed to load device configuration: {error=}")
        return False
    return True


def _save_device_property(
    name: str,
    component: Optional[str] = None,
    component_id: Optional[str] = None,
    value=None,
    **kwargs,
) -> bool:
    property_id = {
        "property_type": PropertyType.DEVICE,
        "name": name,
        "component": component,
        "component_id": component_id,
    }
    property = BackendProperty(
        **property_id,
        value=value,
        source="configuration",
        **kwargs,
    )
    # Check if the property has a template entry:
    # this was the case if it got a timestamp
    ts = BackendProperty.get_timestamp(**property_id)
    if ts is not None:
        property.write()
    else:
        logger.error(
            "Invalid device configuration: "
            f"Property identified by {property_id} "
            "has no valid timestamp, so it has not been "
            "declared in the property template config file.",
            stacklevel=2,
        )
        return False
    return True


# =============================================================================
# Main

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load configuration files")
    parser.add_argument("--device", type=str, required=True)

    args, unknown = parser.parse_known_args()

    device_configuration_file = args.device

    layout = None
    default_values = None

    try:
        device_config = toml.load(device_configuration_file)
        layout = device_config.get("layout")
        # The section's name is the same as the property type's str value:
        default_values = device_config.get(str(PropertyType.DEVICE))
    except Exception as err:
        logger.error(
            f"Failed to load configuration from {device_configuration_file}: {err=}"
        )
        exit(1)

    if not (layout and load_device_layout_configuration(layout)):
        logger.error(
            f"Failed to load layout configuration from "
            f"{device_configuration_file}: "
        )
        exit(1)
    # Update some properties with meta information before loading the
    # remaining configuration
    if not initialize_properties():
        logger.error(f"Failed to initialize backend properties")
        exit(1)
    if default_values:
        if not load_device_configuration(default_values):
            logger.error(
                f"Failed to load additional backend properties "
                f"from {device_configuration_file}"
            )
            exit(1)

    if unknown:
        # Unrecognized arguments, can be passed back to caller if wanted:
        logger.warning(f"Unrecognized arguments: {unknown}")
        print(f"{' '.join(unknown)}")
