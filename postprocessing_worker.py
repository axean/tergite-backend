from pathlib import Path
import time
from uuid import uuid4
from starlette.config import Config

# config
config = Config(".env")
NAME = config("NAME", default="NO-NAME")
DB_URL = config("DB_URL", default="NO-DB-URL")
PREFIX = config("PREFIX", default="pingu")
API_PREFIX = config("API_PREFIX", default=PREFIX)
STORAGE_PREFIX_DIRNAME = config("STORAGE_PREFIX_DIRNAME", default=PREFIX)
STORAGE_ROOT = config("STORAGE_ROOT", default="/tmp")
JOB_UPLOAD_POOL_DIRNAME = config("JOB_UPLOAD_POOL_DIRNAME", default="job_upload_pool")
JOB_EXECUTION_POOL_DIRNAME = config(
    "JOB_EXECUTION_POOL_DIRNAME", default="job_execution_pool"
)
LOGFILE_DOWNLOAD_POOL_DIRNAME = config(
    "LOGFILE_DOWNLOAD_POOL_DIRNAME", default="logfile_download_pool"
)


def logfile_postprocess(logfile: Path):

    print(f"Postprocessing logfile {str(logfile)}")

    # mimick logfile post-processing
    time.sleep(2)

    new_file_name = str(uuid4())
    storage_location = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME

    new_file_path = storage_location / LOGFILE_DOWNLOAD_POOL_DIRNAME
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name

    logfile.replace(new_file)

    print(f"Created new file {str(new_file)}")
