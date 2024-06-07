import json
from os import path
from typing import Any, Dict, List, Union

import yaml

_TESTS_FOLDER = path.dirname(path.dirname(path.abspath(__file__)))
_FIXTURES_PATH = path.join(_TESTS_FOLDER, "fixtures")


def load_fixture(
    file_name: str, fmt: str = "json"
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Loads the given fixture from the fixtures directory

    Args:
        file_name: the name of the file that contains the fixture
        fmt: file format e.g. json or yaml, default: json

    Returns:
        the list of dicts or the dict got from the json fixture file
    """
    fixture_path = get_fixture_path(file_name)
    with open(fixture_path, "rb") as file:
        if fmt == "json":
            return json.load(file)
        elif fmt == "yaml":
            return yaml.load(file, Loader=yaml.FullLoader)
        else:
            raise NotImplementedError(f"Cannot load fixture of format:{fmt}")


def get_fixture_path(*paths: str) -> str:
    """Gets the path to the fixture

    Args:
        paths: sections of paths to the given fixture e.g. fixtures/api/rest/rng_list.json
            would be "api", "rest", "rng_list.json"

    Returns:
        the path to the given fixture
    """
    return path.join(_FIXTURES_PATH, *paths)
