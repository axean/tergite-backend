"""Utility functions for dealing with modules"""
import sys
from typing import List


def remove_modules(module_names: List[str]):
    """Removes all modules whose names start with the given list of modules

    Args:
        module_names: names of modules to remove
    """
    for key in list(sys.modules.keys()):
        if any(key.startswith(name) for name in module_names):
            del sys.modules[key]
