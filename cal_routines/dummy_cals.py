from scenario_scripts import qobj_dummy_scenario
from examples.generate_jobs.qobj_stub_single import generate_job
from uuid import uuid4


def cal_dummy():
    job_id = uuid4()
    job = generate_job()
    scenario = qobj_dummy_scenario(job)
    scenario.tags.tags = [job_id, "calibration", 'dummy']
    return scenario
