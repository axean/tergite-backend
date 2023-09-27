import importlib
import typing

import settings


def import_labber() -> typing.Any:
    """
    This function imports labber only if the respective environmental variable is set
    """
    if settings.IMPORT_LABBER:
        return importlib.import_module('Labber')
    else:
        return importlib.import_module('.LabberDummy')
