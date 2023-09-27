from redis import Redis
from rq import Queue, Worker

from execution_worker import job_execute

import settings

redis = Redis()
queue_name = 'job_execution'
worker_name = f'{settings.DEFAULT_PREFIX}_{queue_name}_worker'

queue = Queue(connection=redis, name=f'{settings.DEFAULT_PREFIX}_{queue_name}')
worker = Worker([queue], connection=redis, name=worker_name)
worker.work()
