"""Utilities for HTTP related stuff"""
from typing import Callable

import httpx
import requests


class MockHttpResponse(httpx.Response):
    """An extension of the httpx.Response"""

    def ok(self):
        return self.is_success


class MockHttpSession(requests.Session):
    """A mock of the requests.Session"""

    def __init__(self, **mock_requests: Callable):
        """
        Args:
            mock_requests: mock requests e.g. `put=put_func, get=get_func`
        """
        super().__init__()
        self._mock_requests = mock_requests

    def get(self, url, **kwargs):
        if "get" in self._mock_requests:
            return self._mock_requests["get"](url, **kwargs)
        return super().get(url, **kwargs)

    def post(self, url, **kwargs):
        if "post" in self._mock_requests:
            return self._mock_requests["post"](url, **kwargs)
        return super().post(url, **kwargs)

    def put(self, url, **kwargs):
        if "put" in self._mock_requests:
            return self._mock_requests["put"](url, **kwargs)
        return super().put(url, **kwargs)

    def delete(self, url, **kwargs):
        if "delete" in self._mock_requests:
            return self._mock_requests["delete"](url, **kwargs)
        return super().delete(url, **kwargs)

    def patch(self, url, **kwargs):
        if "patch" in self._mock_requests:
            return self._mock_requests["patch"](url, **kwargs)
        return super().patch(url, **kwargs)

    def head(self, url, **kwargs):
        if "head" in self._mock_requests:
            return self._mock_requests["head"](url, **kwargs)
        return super().head(url, **kwargs)

    def options(self, url, **kwargs):
        if "options" in self._mock_requests:
            return self._mock_requests["options"](url, **kwargs)
        return super().options(url, **kwargs)
