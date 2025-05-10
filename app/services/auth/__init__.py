"""Entry point for the auth service"""
from .dtos import AuthLog, Credentials, PartialAuthLog
from .exc import AuthenticationError, AuthorizationError, CredentialsAlreadyExists
from .service import authenticate, save_credentials

__all__ = [
    save_credentials,
    authenticate,
    Credentials,
    AuthLog,
    PartialAuthLog,
    CredentialsAlreadyExists,
    AuthorizationError,
    AuthenticationError,
]
