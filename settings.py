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

import redis
from starlette.config import Config
from starlette.datastructures import URL

# NOTE: shell env variables take precedence over the configuration file
env_file = os.environ.get("ENV_FILE", default=".env")
config = Config(Path(__file__).parent / env_file, environ=os.environ)

# Automatic root directory settings
APP_ROOT_DIR = Path(__file__).parent / "app"

# Misc settings
APP_SETTINGS = config("APP_SETTINGS", cast=str, default="production")
IS_AUTH_ENABLED = config("IS_AUTH_ENABLED", cast=bool, default=True)
IS_STANDALONE = config("IS_STANDALONE", cast=bool, default=False)
_is_production = APP_SETTINGS == "production"

if not IS_AUTH_ENABLED and _is_production:
    raise ValueError(
        "'IS_AUTH_ENABLED' environment variable has been set to false in production."
    )

# Storage settings

DEFAULT_PREFIX = config("DEFAULT_PREFIX", cast=str)
STORAGE_ROOT = config("STORAGE_ROOT", cast=str, default="/tmp")
STORAGE_PREFIX_DIRNAME = config(
    "STORAGE_PREFIX_DIRNAME", cast=str, default=DEFAULT_PREFIX
)

LOGFILE_DOWNLOAD_POOL_DIRNAME = config(
    "LOGFILE_DOWNLOAD_POOL_DIRNAME", cast=str, default="logfile_download_pool"
)
LOGFILE_UPLOAD_POOL_DIRNAME = config(
    "LOGFILE_UPLOAD_POOL_DIRNAME", cast=str, default="logfile_upload_pool"
)
JOB_UPLOAD_POOL_DIRNAME = config(
    "JOB_UPLOAD_POOL_DIRNAME", cast=str, default="job_upload_pool"
)
JOB_PRE_PROC_POOL_DIRNAME = config(
    "JOB_PRE_PROC_POOL_DIRNAME", cast=str, default="job_preproc_pool"
)
JOB_EXECUTION_POOL_DIRNAME = config(
    "JOB_EXECUTION_POOL_DIRNAME", cast=str, default="job_execution_pool"
)
JOB_SUPERVISOR_LOG = config(
    "JOB_SUPERVISOR_LOG", cast=str, default="job_supervisor.log"
)
EXECUTOR_DATA_DIRNAME = config(
    "EXECUTOR_DATA_DIRNAME", cast=str, default="executor_data"
)

_executor_data_dir_path = os.path.join(
    STORAGE_ROOT, DEFAULT_PREFIX, EXECUTOR_DATA_DIRNAME
)
if not os.path.exists(_executor_data_dir_path):
    os.makedirs(_executor_data_dir_path)
EXECUTOR_DATA_DIR = _executor_data_dir_path

# Measurement default file mapping
MEASUREMENT_DEFAULT_FILES = config(
    "MEASUREMENT_DEFAULT_FILES",
    cast=str,
    default="measurement_jobs/parameter_defaults/default_files.toml",
)

# Definition of backend property names
BACKEND_SETTINGS = config(
    "BACKEND_SETTINGS",
    cast=str,
    default=Path(__file__).parent / "backend_config.toml",
)

# Connectivity settings
MSS_MACHINE_ROOT_URL = config(
    "MSS_MACHINE_ROOT_URL", cast=URL, default="http://localhost:8002"
)
BCC_MACHINE_ROOT_URL = config(
    "BCC_MACHINE_ROOT_URL", cast=URL, default="http://localhost:8000"
)
BCC_PORT = config("BCC_PORT", cast=int, default=8000)

# Authentication

MSS_APP_TOKEN = config("MSS_APP_TOKEN", cast=str, default="")

# BCC should be hidden from the internet except for a few endpoints, and IPs
# WhiteLIST is actually a dict to ensure O(1) lookup time everytime
CLIENT_IP_WHITELIST = {
    socket.gethostbyname(v.hostname): True
    for v in [
        MSS_MACHINE_ROOT_URL,
    ]
}
# allow test client to access api when BLACKLISTED is not set
if APP_SETTINGS == "test" and not os.environ.get("BLACKLISTED"):
    CLIENT_IP_WHITELIST["testclient"] = True

# -----------------------
# Hardware configurations
# -----------------------
# The executor type specifies which implementations of the QuantumExecutor to use.
# For more information on the values check:
# - dot-env-template.txt
EXECUTOR_TYPE = config("EXECUTOR_TYPE", default="quantify")

# This will load the hardware configuration from a yaml file, which contains the properties for the
# cluster or other instrument setup. For more information check:
# - quantify-config.example.yml
QUANTIFY_CONFIG_FILE = config("QUANTIFY_CONFIG_FILE", default="quantify-config.yml")

# -------------
# Redis config
# -------------
REDIS_HOST = config("REDIS_HOST", default="localhost")
REDIS_PORT = config("REDIS_PORT", default=6379)
REDIS_USER = config("REDIS_USER", default=None)
REDIS_PASSWORD = config("REDIS_PASSWORD", default=None)
REDIS_DB = config("REDIS_DB", cast=int, default=0)

# For convenience to import globally
REDIS_CONNECTION = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    username=REDIS_USER,
    password=REDIS_PASSWORD,
)
