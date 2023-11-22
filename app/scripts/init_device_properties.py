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
#
# Modified:
#
# - Martin Ahindura 2023

from typing import List, Optional

import toml

import settings

from ..utils.logging import get_logger
from ..utils.storage import BackendProperty, PropertyType, set_component_property

"""Logging initialization"""

logger = get_logger()


# =============================================================================
# Create empty Redis entries with pre-defined backend properties

# Question: where should this filepath be defined? .env?
# For testing purposes it is good if this path can be configured.
TEMPLATE_FILE = settings.BACKEND_PROPERTIES_TEMPLATE


# precondition: device layout configuration has been loaded,
# which means: we know which components are present, and in what numbers
def initialize_properties() -> bool:
    try:
        property_template = toml.load(TEMPLATE_FILE)
        # The section's name is the same as the property type's str value:
        device_template = property_template[str(PropertyType.DEVICE)]
        components = BackendProperty.read_value(
            property_type=PropertyType.DEVICE, name="components"
        )
        for tag, tag_dict in device_template.items():
            if tag in components:  # then tag is a component property
                for name, fields in tag_dict.items():
                    components_ids = get_component_ids(tag)
                    # Populate Redis from template for each component id
                    for component_id in components_ids:
                        set_component_property(
                            name=name,
                            component=tag,
                            component_id=component_id,
                            value=None,  # serves as a placeholder in Redis
                            source="default",
                            **fields,
                        )
            else:
                # otherwise tag is a device property name
                BackendProperty(
                    property_type=PropertyType.DEVICE,
                    name=tag,
                    **tag_dict,
                ).write()
    except Exception as error:
        logger.error(f"Failed to initialize properties: {error=}")
        return False
    return True


def set_component_ids(component: str, ids: List[str]):
    property_name = f"{component}_ids"
    BackendProperty(
        property_type=PropertyType.DEVICE,
        name=property_name,
        value=ids,
        source="configuration",
    ).write()


def get_component_ids(component: str) -> Optional[List[str]]:
    property_name = f"{component}_ids"
    return BackendProperty.read_value(
        property_type=PropertyType.DEVICE, name=property_name
    )


# =============================================================================
# Main

if __name__ == "__main__":
    # Requires number_of_resonators to be set in Redis
    initialize_properties()
