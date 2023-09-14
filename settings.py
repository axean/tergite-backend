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
from starlette.datastructures import URL, CommaSeparatedStrings

# NOTE: shell env variables take precedence over the configuration file
config = Config(".env")


# Misc settings

# Plotting during post-processing, only for interactive use, *not*
# when running as a server
POSTPROC_PLOTTING = config("POSTPROC_PLOTTING", cast=bool, default=False)


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
# See also backend_properties_config/device_*.toml
BACKEND_PROPERTIES_TEMPLATE = config(
    "BACKEND_PROPERTIES_TEMPLATE",
    cast=str,
    default="backend_properties_config/property_templates_default.toml",
)

# Connectivity settings

LABBER_MACHINE_ROOT_URL = config("LABBER_MACHINE_ROOT_URL", cast=URL)
QUANTIFY_MACHINE_ROOT_URL = config("QUANTIFY_MACHINE_ROOT_URL", cast=URL)
MSS_MACHINE_ROOT_URL = config("MSS_MACHINE_ROOT_URL", cast=URL)
BCC_MACHINE_ROOT_URL = config("BCC_MACHINE_ROOT_URL", cast=URL)
BCC_PORT = config("BCC_PORT", cast=int)
DB_MACHINE_ROOT_URL = config("DB_MACHINE_ROOT_URL", cast=URL)

CALIBRATION_SUPERVISOR_PORT = config("CALIBRATION_SUPERVISOR_PORT", cast=int)

# Calibration supervisor settings

CALIBRATION_GRAPH = config(
    "CALIBRATION_GRAPH", cast=str, default="calibration_graphs/default.json"
)
CALIBRATION_GOALS = list(
    config("CALIBRATION_GOALS", cast=CommaSeparatedStrings, default=[])
)
