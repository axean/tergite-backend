"""Utilities for datetime"""

from datetime import datetime, timezone


def utc_now_str() -> str:
    """Returns current time in UTC string but with hours replaced with a Z"""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
