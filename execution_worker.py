import pathlib
import json
import time


def execute_job(file):
    print(f"Executing file {str(file)}")

    job_dict = {}
    with file.open() as f:
        job_dict = json.load(f)

    print(f"Command: {job_dict['name']}")
    time.sleep(3)

    file.unlink()
