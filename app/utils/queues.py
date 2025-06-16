# This code is part of Tergite
#
# (C) Copyright Martin Ahindura 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for handling queues"""
from redis import Redis
from rq import Queue


class QueuePool:
    """A collection of Queues relevant to BCC

    Attributes:
        job_registration_queue: the queue for handling job registration
        logfile_postprocessing_queue: the queue for handling logfile postprocessing
        job_execution_queue: the queue for handling job execution
        connection: the redis connection where the queue pool is to run
        is_async: whether the queues are to be run asynchronously or not
    """

    def __init__(self, prefix: str, connection: "Redis", is_async: bool = True):
        """
        Args:
            prefix: the prefix for the names of the expected queues
            connection: the connection to Redis
            is_async: whether to dispatch the enqueued tasks in other workers
        """
        self.connection = connection
        self.is_async = is_async

        self.job_registration_queue = Queue(
            f"{prefix}_job_registration", connection=connection, is_async=is_async
        )
        self.logfile_postprocessing_queue = Queue(
            f"{prefix}_logfile_postprocessing", connection=connection, is_async=is_async
        )
        self.job_execution_queue = Queue(
            f"{prefix}_job_execution", connection=connection, is_async=is_async
        )
