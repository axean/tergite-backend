"""Utilities for HTTP related stuff"""
import httpx


class MockHttpResponse(httpx.Response):
    """An extension of the httpx.Response"""

    def ok(self):
        return self.is_success
