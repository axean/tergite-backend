from typing import Dict

from .utils.env import (
    TEST_DEFAULT_PREFIX,
    TEST_LABBER_MACHINE_ROOT_URL,
    TEST_LOGFILE_DOWNLOAD_POOL_DIRNAME,
    TEST_MSS_MACHINE_ROOT_URL,
    TEST_QUANTIFY_MACHINE_ROOT_URL,
    TEST_STORAGE_PREFIX_DIRNAME,
    TEST_STORAGE_ROOT,
    setup_test_env,
)

# set up the environment before any other import
setup_test_env()

import shutil
from pathlib import Path

import pytest
from fakeredis import FakeStrictRedis
from fastapi.testclient import TestClient
from freezegun import freeze_time
from pytest_lazyfixture import lazy_fixture
from redis.client import Redis
from rq import SimpleWorker

from ..utils.queues import QueuePool
from .utils.fixtures import load_json_fixture
from .utils.http import MockHttpResponse, MockHttpSession
from .utils.modules import remove_modules
from .utils.rq import get_rq_worker

_lda_parameters_fixture = load_json_fixture("lda_parameters.json")
_real_redis = Redis(db=2)
_fake_redis = FakeStrictRedis()
_async_queue_pool = QueuePool(
    prefix=TEST_DEFAULT_PREFIX, connection=_real_redis, is_async=True
)
_sync_queue_pool = QueuePool(
    prefix=TEST_DEFAULT_PREFIX, connection=_fake_redis, is_async=False
)

MOCK_NOW = "2023-11-27T12:46:48.851656+00:00"
TEST_APP_TOKEN_STRING = "eecbf107ad103f70187923f49c1a1141219da95f1ab3906f"

FASTAPI_CLIENTS = [
    lazy_fixture("async_fastapi_client"),
    # FIXME: See issue https://bitbucket.org/qtlteam/tergite-bcc/issues/1/inform-job-location-stage-logic-is
    # lazy_fixture("sync_fastapi_client"),
]

CLIENTS = [
    (lazy_fixture("async_fastapi_client"), lazy_fixture("real_redis_client")),
    # FIXME: See issue https://bitbucket.org/qtlteam/tergite-bcc/issues/1/inform-job-location-stage-logic-is
    # (lazy_fixture("sync_fastapi_client"), lazy_fixture("fake_redis_client")),
]

CLIENT_AND_RQ_WORKER_TUPLES = [
    (
        lazy_fixture("async_fastapi_client"),
        lazy_fixture("real_redis_client"),
        lazy_fixture("async_rq_worker"),
    ),
    # FIXME: See issue https://bitbucket.org/qtlteam/tergite-bcc/issues/1/inform-job-location-stage-logic-is
    # (
    #     lazy_fixture("sync_fastapi_client"),
    #     lazy_fixture("fake_redis_client"),
    #     lazy_fixture("sync_rq_worker"),
    # ),
]


def mock_post_requests(url: str, **kwargs):
    """Mock POST requests for testing"""
    if url == f"{TEST_QUANTIFY_MACHINE_ROOT_URL}/qobj":
        return MockHttpResponse(status_code=200)
    if url == f"{TEST_LABBER_MACHINE_ROOT_URL}/scenarios":
        return MockHttpResponse(status_code=200)
    if url == f"{TEST_QUANTIFY_MACHINE_ROOT_URL}/rng_LokiB":
        return MockHttpResponse(status_code=200)


def mock_mss_get_requests(url: str, **kwargs):
    """Mock GET requests sent to MSS for testing"""
    if url.endswith("properties/lda_parameters"):
        return MockHttpResponse(status_code=200, json=_lda_parameters_fixture)


def mock_mss_put_requests(url: str, **kwargs):
    """Mock PUT requests sent to MSS for testing"""
    if url.startswith(f"{TEST_MSS_MACHINE_ROOT_URL}/jobs"):
        return MockHttpResponse(status_code=200)


@pytest.fixture
def real_redis_client() -> Redis:
    """A mock redis client"""
    yield _real_redis
    _real_redis.flushall()


@pytest.fixture
def fake_redis_client() -> Redis:
    """A mock redis client"""
    yield _fake_redis
    _fake_redis.flushall()


@pytest.fixture
def async_rq_worker() -> SimpleWorker:
    """Get the rq worker for running async tasks asynchronously"""
    yield get_rq_worker(_async_queue_pool)


@pytest.fixture
def sync_rq_worker() -> SimpleWorker:
    """Get the rq worker for running tasks synchronously"""
    yield get_rq_worker(_sync_queue_pool)


@pytest.fixture
def async_fastapi_client(mocker) -> TestClient:
    """A test client for fast api when rq is running asynchronously"""
    remove_modules(["app"])

    mss_client = MockHttpSession(put=mock_mss_put_requests, get=mock_mss_get_requests)

    mocker.patch("redis.Redis", return_value=_real_redis)
    mocker.patch("app.utils.queues.QueuePool", return_value=_async_queue_pool)
    mocker.patch("requests.post", side_effect=mock_post_requests)
    mocker.patch("requests.Session", return_value=mss_client)

    from app.api import app

    with freeze_time(MOCK_NOW):
        yield TestClient(app)


@pytest.fixture
def sync_fastapi_client(mocker) -> TestClient:
    """A test client for fast api when rq is running asynchronously"""
    remove_modules(["app"])

    mss_client = MockHttpSession(put=mock_mss_put_requests, get=mock_mss_get_requests)

    mocker.patch("redis.Redis", return_value=_fake_redis)
    mocker.patch("app.utils.queues.QueuePool", return_value=_sync_queue_pool)
    mocker.patch("requests.post", side_effect=mock_post_requests)
    mocker.patch("requests.Session", return_value=mss_client)

    from app.api import app

    with freeze_time(MOCK_NOW):
        client = TestClient(app)
        client.app.state.mss_client = mss_client
        yield client


@pytest.fixture
def client_jobs_folder() -> Path:
    """A temporary folder for the client where jobs can be saved"""
    folder_path = Path("./tmp/jobs")
    folder_path.mkdir(parents=True, exist_ok=True)

    yield folder_path
    shutil.rmtree(folder_path, ignore_errors=True)


@pytest.fixture
def logfile_download_folder() -> Path:
    """A temporary folder for the server where logfiles can be downloaded from"""
    folder_path = (
        Path(TEST_STORAGE_ROOT)
        / TEST_STORAGE_PREFIX_DIRNAME
        / TEST_LOGFILE_DOWNLOAD_POOL_DIRNAME
    )
    folder_path.mkdir(parents=True, exist_ok=True)

    yield folder_path
    shutil.rmtree(folder_path, ignore_errors=True)


@pytest.fixture
def app_token_header() -> Dict[str, str]:
    """the authorization header with the app token"""

    yield {"Authorization": f"Bearer {TEST_APP_TOKEN_STRING}"}
