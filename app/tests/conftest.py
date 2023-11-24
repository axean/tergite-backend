import sys

from redis.client import Redis

from .utils.env import (
    TEST_DEFAULT_PREFIX,
    TEST_LABBER_MACHINE_ROOT_URL,
    TEST_QUANTIFY_MACHINE_ROOT_URL,
    setup_test_env,
)
from .utils.rq import WindowsSimpleWorker

# set up the environment before any other import
setup_test_env()

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rq import SimpleWorker

from ..utils.queues import QueuePool
from .utils.http import MockHttpResponse

_mock_redis = Redis(db=2)
_mock_queue_pool = QueuePool(
    prefix=TEST_DEFAULT_PREFIX, connection=_mock_redis, is_async=True
)


def mock_post_requests(url: str, **kwargs):
    """Mock post requests for testing"""
    if url == f"{TEST_QUANTIFY_MACHINE_ROOT_URL}/qobj":
        return MockHttpResponse(status_code=200)
    if url == f"{TEST_LABBER_MACHINE_ROOT_URL}/scenarios":
        return MockHttpResponse(status_code=200)


@pytest.fixture
def redis_client() -> Redis:
    """A mock redis client"""
    yield _mock_redis
    _mock_redis.flushall()


@pytest.fixture
def queue_pool() -> QueuePool:
    """A mock QueuePool"""
    yield _mock_queue_pool


@pytest.fixture
def rq_worker(queue_pool: QueuePool) -> SimpleWorker:
    """Get the rq worker for running async tasks"""
    if sys.platform.startswith("win32"):
        return WindowsSimpleWorker(
            [
                queue_pool.job_registration_queue,
                queue_pool.logfile_postprocessing_queue,
                queue_pool.job_execution_queue,
                queue_pool.job_preprocessing_queue,
            ],
            connection=queue_pool.connection,
        )
    return SimpleWorker(
        [
            queue_pool.job_registration_queue,
            queue_pool.logfile_postprocessing_queue,
            queue_pool.job_execution_queue,
            queue_pool.job_preprocessing_queue,
        ],
        connection=queue_pool.connection,
    )


@pytest.fixture
def external_services(mocker):
    """External services like MSS, Quantify connector and Labber machine"""
    mocker.patch("redis.Redis", return_value=_mock_redis)
    mocker.patch("app.utils.queues.QueuePool", return_value=_mock_queue_pool)
    mocker.patch("requests.post", side_effect=mock_post_requests)


@pytest.fixture
def client(external_services) -> TestClient:
    """A test client for fast api"""
    from app.api import app

    yield TestClient(app)


@pytest.fixture
def client_jobs_folder() -> Path:
    """A temporary folder for the client where jobs can be saved"""
    folder_path = Path("./tmp/jobs")
    folder_path.mkdir(parents=True, exist_ok=True)

    yield folder_path
    shutil.rmtree(folder_path, ignore_errors=True)
