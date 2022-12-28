import json
from pathlib import Path

import requests

import settings

STORAGE_ROOT = settings.STORAGE_ROOT
QUANTIFY_MACHINE_ROOT_URL = settings.QUANTIFY_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
MSS_MACHINE_ROOT_URL = settings.MSS_MACHINE_ROOT_URL

REST_API_MAP = {"rng_LokiB": "/rng_LokiB"}


def quantify_rng(job_id):
    tmp_file = Path(STORAGE_ROOT) / (str(job_id) + ".to_quantify")
    job_dict = {"job_id": str(job_id)}

    with tmp_file.open("w") as store:
        json.dump(job_dict, store)  # copy incoming data to temporary file

    with tmp_file.open("r") as source:
        files = {
            "upload_file": (tmp_file.name, source),
            "mss_url": (None, str(MSS_MACHINE_ROOT_URL)),
        }
        url = str(QUANTIFY_MACHINE_ROOT_URL) + REST_API_MAP["rng_LokiB"]
        response = requests.post(url, files=files)
    print("Asking QBLOX for random numbers..")
