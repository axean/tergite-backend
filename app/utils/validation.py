# This code is part of Tergite
#
# (C) Stefan Hill (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json
from typing import Mapping, Any

from settings import APP_ROOT_DIR

from jsonschema import validate, ValidationError, SchemaError


def _load_json_schema_definition(schema_definition_name: str) -> Mapping[str, Any]:
    """
    Loads a JSON schema definition from the resources

    Args:
        schema_definition_name: Name of the JSON file to load (without .json suffix)

    Returns:
        JSON object of the schema definition

    """
    filepath = APP_ROOT_DIR / f"resources/schema/json/{schema_definition_name}.json"
    with open(filepath, "r") as file:
        schema_definition = json.load(file)
    return schema_definition


def validate_schema(json_obj: Mapping[str, Any],
                    schema_definition_name: str) -> bool:
    """
    Validates a JSON object against a predefined schema definition

    Args:
        json_obj: JSON object to be validated
        schema_definition_name: Name of the predefined JSON schema definition to load
                                Schema definitions are in app/resources/schema/json

    Returns:
        True or false whether the schemas match, catches common errors, but prints a warning

    """

    # FIXME: How do we do in general logging in the backend?

    schema_definition = _load_json_schema_definition(schema_definition_name)
    try:
        validate(instance=json_obj, schema=schema_definition)
        return True
    except ValidationError as e:
        print("JSON object is invalid:", e.message)
        return False
    except SchemaError as e:
        print("Schema is invalid:", e.message)
        return False
