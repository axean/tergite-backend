from starlette.config import Config
from redis import Redis
from rq import Queue, Worker
import shutil
import pathlib
from uuid import uuid4
import time
from execution_worker import execute_job


# config
config = Config(".env")
NAME = config("NAME", default="NO-NAME")
DB_URL = config("DB_URL", default="NO-DB-URL")
PREFIX = config("PREFIX", default="pingu")
API_PREFIX = config("API_PREFIX", default=PREFIX)
STORAGE_PREFIX_DIRNAME = config("STORAGE_PREFIX_DIRNAME", default=PREFIX)
STORAGE_ROOT = config("STORAGE_ROOT", default="/tmp")
UPLOAD_POOL_DIRNAME = config("UPLOAD_POOL_DIRNAME", default="upload_pool")
EXECUTION_POOL_DIRNAME = config("EXECUTION_POOL_DIRNAME", default="execution_pool")

# redis connection
redis_connection = Redis()

rq_job_execution = Queue(PREFIX + "_job_execution", connection=redis_connection)


def preprocess(file):

    print(f"Preprocessing file {str(file)}")

    # model pre-processing
    time.sleep(2)

    new_file_name = str(uuid4())
    storage_location = pathlib.Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME
    new_file_path = storage_location / EXECUTION_POOL_DIRNAME
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name

    file.replace(new_file)

    rq_job_execution.enqueue(execute_job, new_file)

    print(f"Created new file {str(new_file)}")
