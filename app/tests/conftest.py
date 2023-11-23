from .utils.env import setup_test_env

# set up the environment before any other import
setup_test_env()

import pytest
from fakeredis import FakeStrictRedis
from fastapi.testclient import TestClient

_mock_redis = FakeStrictRedis()


@pytest.fixture
def redis_client(mocker) -> FakeStrictRedis:
    """A mock redis client"""
    mocker.patch("redis.Redis", return_value=_mock_redis)
    yield _mock_redis
    _mock_redis.flushall()


@pytest.fixture
def client(redis_client) -> TestClient:
    """A test client for fast api"""
    from app.api import app

    yield TestClient(app)
