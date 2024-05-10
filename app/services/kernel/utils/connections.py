# This code is part of Tergite
#
# (C) Martin Ahindura (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for connections to kernel service"""
from multiprocessing.connection import Connection
from typing import Any


def receive_msg(channel: Connection, timeout: float = 2) -> Any:
    """Waits for a message from the connection or times out if none is received in `timeout` seconds

    Args:
        channel: the connection to send the message on
        timeout: the timeout in seconds. default = 2

    Returns:
        the message from connection

    Raises:
        TimeoutError: no message received in {timeout} seconds
    """
    if channel.poll(timeout=timeout):
        return channel.recv()
    raise TimeoutError(f"no message received in {timeout} seconds")
