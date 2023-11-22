from redis import Redis
from rq import Queue, Worker

import multiprocessing

import settings

WORKER_QUEUES = ['job_registration',
                 'job_preprocessing',
                 'job_execution',
                 'logfile_postprocessing']

# WORKER_QUEUES = ['logfile_postprocessing']

redis = Redis()


def _run_rq_worker(worker_name_: str,
                   worker_queue_: str,
                   connection_: 'Redis'):
    # Establish a connection to Redis
    queue = Queue(connection=connection_, name=worker_queue_)
    worker = Worker([queue], connection=connection_, name=worker_name_)
    worker.work()


if __name__ == '__main__':
    # Create a separate process for the RQ worker

    processes = {}

    for i, queue in enumerate(WORKER_QUEUES):
        worker_name = f'{settings.DEFAULT_PREFIX}_{queue}_worker'
        processes[i] = multiprocessing.Process(target=_run_rq_worker,
                                               args=(worker_name,
                                                     f'{settings.DEFAULT_PREFIX}_{queue}',
                                                     redis))

        # Start the RQ worker process
        processes[i].start()

    for i in range(len(WORKER_QUEUES)):
        try:
            # Wait for the RQ worker process to complete
            processes[i].join()
        except KeyboardInterrupt:
            # Handle keyboard interrupt (e.g., Ctrl+C) if needed
            print("Terminating RQ worker process...")
            processes[i].terminate()
            processes[i].join()

        print("RQ worker process has exited.")
