import pytest

from app.tests.utils.fixtures import load_json_fixture
from app.tests.utils.redis import insert_in_hash

_JOBS_LIST = load_json_fixture("job_list.json")
_JOB_ID_FIELD = "job_id"
_JOB_IDS = [item[_JOB_ID_FIELD] for item in _JOBS_LIST]

_JOBS_HASH_NAME = "job_supervisor"


def test_root(client):
    """GET / returns "message": "Welcome to BCC machine"""
    with client as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to BCC machine"}


def test_fetch_all_jobs(redis_client, client):
    """Get to /jobs returns tall jobs"""
    insert_in_hash(
        client=redis_client,
        hash_name=_JOBS_HASH_NAME,
        data=_JOBS_LIST,
        id_field=_JOB_ID_FIELD,
    )

    # using context manager to ensure on_startup runs
    with client as client:
        response = client.get("/jobs")
        got = response.json()
        expected = {item[_JOB_ID_FIELD]: item for item in _JOBS_LIST}

        assert response.status_code == 200
        assert got == expected


@pytest.mark.parametrize("job_id", _JOB_IDS)
def test_fetch_job(redis_client, client, job_id: str):
    """Get to /jobs/{job_id} returns the job for the given job_id"""
    insert_in_hash(
        client=redis_client,
        hash_name=_JOBS_HASH_NAME,
        data=_JOBS_LIST,
        id_field=_JOB_ID_FIELD,
    )

    # using context manager to ensure on_startup runs
    with client as client:
        response = client.get(f"/jobs/{job_id}")
        got = response.json()
        expected = {
            "message": list(filter(lambda x: x["job_id"] == job_id, _JOBS_LIST))[0]
        }

        assert response.status_code == 200
        assert got == expected


@pytest.mark.parametrize("job_id", _JOB_IDS)
def test_fetch_job_result(redis_client, client, job_id: str):
    """Get to /jobs/{job_id}/result returns the job result for the given job_id"""
    insert_in_hash(
        client=redis_client,
        hash_name=_JOBS_HASH_NAME,
        data=_JOBS_LIST,
        id_field=_JOB_ID_FIELD,
    )

    # using context manager to ensure on_startup runs
    with client as client:
        response = client.get(f"/jobs/{job_id}/result")
        got = response.json()
        expected_job = list(filter(lambda x: x["job_id"] == job_id, _JOBS_LIST))[0]

        try:
            expected = {"message": expected_job["result"]}
        except KeyError:
            expected = {"message": "job has not finished"}

        assert response.status_code == 200
        assert got == expected


@pytest.mark.parametrize("job_id", _JOB_IDS)
def test_fetch_job_status(redis_client, client, job_id: str):
    """Get to /jobs/{job_id}/status returns the job status for the given job_id"""
    # importing this here so that patching of redis.Redis does not get messed up
    # as it would if the import statement was at the beginning of the file.
    # FIXME: In future, the global `red = redis.Redis()` scattered in the code should be deleted
    from app.services.jobs.service import STR_LOC, Location

    insert_in_hash(
        client=redis_client,
        hash_name=_JOBS_HASH_NAME,
        data=_JOBS_LIST,
        id_field=_JOB_ID_FIELD,
    )

    # using context manager to ensure on_startup runs
    with client as client:
        response = client.get(f"/jobs/{job_id}/status")
        got = response.json()
        expected_job = list(filter(lambda x: x["job_id"] == job_id, _JOBS_LIST))[0]

        try:
            status = expected_job["status"]
            status["location"] = STR_LOC[Location(status["location"])]
            expected = {"message": status}
        except KeyError:
            expected = {"message": f"job {job_id} not found"}

        assert response.status_code == 200
        assert got == expected


# TODO: Add more tests
