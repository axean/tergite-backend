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
        queue_pool.job_preprocessing_queue,
    ]
    if not queue_pool.is_async:
        return PseudoSimpleWorker(queues=queues, connection=queue_pool.connection)

    if sys.platform.startswith("win32"):
        return WindowsSimpleWorker(queues=queues, connection=queue_pool.connection)

    return SimpleWorker(queues=queues, connection=queue_pool.connection)
