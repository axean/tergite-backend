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


from starlette.config import Config
from starlette.datastructures import URL

# NOTE: shell env variables take precedence over the configuration file
config = Config(".env")

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
JOB_SUPERVISOR_LOG = config("JOB_SUPERVISOR_LOG", cast=str)
LABBER_MACHINE_ROOT_URL = config("LABBER_MACHINE_ROOT_URL", cast=URL)
MSS_MACHINE_ROOT_URL = config("MSS_MACHINE_ROOT_URL", cast=URL)
BCC_MACHINE_ROOT_URL = config("BCC_MACHINE_ROOT_URL", cast=URL)
DB_MACHINE_ROOT_URL = config("DB_MACHINE_ROOT_URL", cast=URL)

CALIBRATION_SUPERVISOR_PORT = config("CALIBRATION_SUPERVISOR_PORT", cast=int)
