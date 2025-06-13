# This code is part of Tergite
#
# (C) Copyright Martin Ahindura 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for testing rq workers and queues"""
import sys

from rq import SimpleWorker
from rq.timeouts import TimerDeathPenalty

from app.utils.queues import QueuePool


class WindowsSimpleWorker(SimpleWorker):
    death_penalty_class = TimerDeathPenalty


class PseudoSimpleWorker(SimpleWorker):
    death_penalty_class = TimerDeathPenalty

    def work(self, burst: bool = False, **kwargs) -> bool:
        return True


def get_rq_worker(queue_pool: QueuePool) -> SimpleWorker:
    """Returns an rq worker to run the given queue pool

    Args:
        queue_pool: the QueuePool whose queues are to be run
    """
    queues = [
        queue_pool.job_registration_queue,
        queue_pool.logfile_postprocessing_queue,
        queue_pool.job_execution_queue,
    ]
    if not queue_pool.is_async:
        return PseudoSimpleWorker(queues=queues, connection=queue_pool.connection)

    if sys.platform.startswith("win32"):
        return WindowsSimpleWorker(queues=queues, connection=queue_pool.connection)

    return SimpleWorker(queues=queues, connection=queue_pool.connection)
