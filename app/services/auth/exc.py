"""Exceptions related to the auth service"""


class BaseAuthException(Exception):
    def __init__(self, message: str = ""):
        self._message = message

    def __repr__(self):
        return f"{self.__class__.__name__}: {self._message}"

    def __str__(self):
        return self._message if self._message else self.__class__.__name__


class JobAlreadyExists(BaseAuthException):
    """Exception when new credentials are being saved yet they exist already"""

    pass


class AuthenticationError(BaseAuthException):
    """Exception when credentials passed do not match those given"""

    pass


class AuthorizationError(BaseAuthException):
    """Exception when credentials passed are not allowed to do a certain operation"""

    pass
