# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020
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
# - Stefan Hill, 2024
#

import os
import socket
from pathlib import Path

from starlette.config import Config
from starlette.datastructures import URL, CommaSeparatedStrings

# NOTE: shell env variables take precedence over the configuration file
config = Config(Path(__file__).parent / ".env")

# Misc settings
APP_SETTINGS = config("APP_SETTINGS", cast=str, default="production")
IS_AUTH_ENABLED = config("IS_AUTH_ENABLED", cast=bool, default=True)
_is_production = APP_SETTINGS == "production"

if not IS_AUTH_ENABLED and _is_production:
    raise ValueError(
        "'IS_AUTH_ENABLED' environment variable has been set to false in production."
    )

# Discrimination settings for the simulator
DISCRIMINATE_TWO_STATE = config("DISCRIMINATE_TWO_STATE", cast=bool, default=False)

# Storage settings

DEFAULT_PREFIX = config("DEFAULT_PREFIX", cast=str)
STORAGE_ROOT = config("STORAGE_ROOT", cast=str)
STORAGE_PREFIX_DIRNAME = config(
    "STORAGE_PREFIX_DIRNAME", cast=str, default=DEFAULT_PREFIX
)
LOGFILE_DOWNLOAD_POOL_DIRNAME = config("LOGFILE_DOWNLOAD_POOL_DIRNAME", cast=str)
LOGFILE_UPLOAD_POOL_DIRNAME = config("LOGFILE_UPLOAD_POOL_DIRNAME", cast=str)
JOB_UPLOAD_POOL_DIRNAME = config("JOB_UPLOAD_POOL_DIRNAME", cast=str)
JOB_PRE_PROC_POOL_DIRNAME = config("JOB_PRE_PROC_POOL_DIRNAME", cast=str)
JOB_EXECUTION_POOL_DIRNAME = config("JOB_EXECUTION_POOL_DIRNAME", cast=str)
JOB_SUPERVISOR_LOG = config(
    "JOB_SUPERVISOR_LOG", cast=str, default="job_supervisor.log"
)

# Measurement default file mapping
MEASUREMENT_DEFAULT_FILES = config(
    "MEASUREMENT_DEFAULT_FILES",
    cast=str,
    default="measurement_jobs/parameter_defaults/default_files.toml",
)

# Definition of backend property names
# See also configs/device_*.toml
BACKEND_PROPERTIES_TEMPLATE = config(
    "BACKEND_PROPERTIES_TEMPLATE",
    cast=str,
    default="configs/property_templates_default.toml",
)

BACKEND_SETTINGS = config(
    "BACKEND_SETTINGS",
    cast=str,
    default="configs/backend_config_default.toml",
)

# Connectivity settings

QUANTIFY_MACHINE_ROOT_URL = config("QUANTIFY_MACHINE_ROOT_URL", cast=URL)
MSS_MACHINE_ROOT_URL = config("MSS_MACHINE_ROOT_URL", cast=URL)
BCC_MACHINE_ROOT_URL = config("BCC_MACHINE_ROOT_URL", cast=URL)
BCC_PORT = config("BCC_PORT", cast=int)
DB_MACHINE_ROOT_URL = config("DB_MACHINE_ROOT_URL", cast=URL)

# Authentication

MSS_APP_TOKEN = config("MSS_APP_TOKEN", cast=str, default="")

# BCC should be hidden from the internet except for a few endpoints, and IPs
# WhiteLIST is actually a dict to ensure O(1) lookup time everytime
CLIENT_IP_WHITELIST = {
    socket.gethostbyname(v.hostname): True
    for v in [
        MSS_MACHINE_ROOT_URL,
        QUANTIFY_MACHINE_ROOT_URL,
    ]
}
# allow test client to access api when BLACKLISTED is not set
if APP_SETTINGS == "test" and not os.environ["BLACKLISTED"]:
    CLIENT_IP_WHITELIST["testclient"] = True
