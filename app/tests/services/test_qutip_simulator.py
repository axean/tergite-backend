# global variables
from datetime import datetime
from os import environ

import pytest

from app.services.quantum_executor.utils.serialization import iqx_rld

environ["DATA_DIR"] = "/Users/stefanhi/repos/tergite-backend/app/tests/pytest-datadir"
# environ['EXECUTOR_CONFIG_FILE'] = '/Users/stefanhi/repos/tergite-bcc/app/tests/fixtures/simulator_backend.yml'

from qiskit.providers.ibmq.utils import json_decoder
from qiskit.qobj import PulseQobj

from ...services.quantum_executor.executors.qutip import QuTipExecutor
from app.tests.utils.fixtures import load_json_fixture

connector = QuTipExecutor(
    config_file="/Users/stefanhi/repos/tergite-bcc/app/tests/fixtures/simulator_backend.yml"
)


@pytest.mark.skip
def test_transpile_x_gate_simulator():
    # Load job with x gate from the json
    job_dict = load_json_fixture("x_gate.json")

    # Convert it to an experiment
    assert True


@pytest.mark.skip
def test_job_transpile():
    job_dict = load_json_fixture("y_gate.json")

    job_id = job_dict["job_id"]
    qobj = job_dict["params"]["qobj"]

    if "tag" in qobj["header"].keys():
        connector.register_job(qobj["header"]["tag"])
    else:
        connector.register_job("")

    # --- RLD pulse library
    # [([a,b], 2),...] -> [[a,b],[a,b],...]
    for pulse in qobj["config"]["pulse_library"]:
        pulse["samples"] = iqx_rld(pulse["samples"])

    # --- In-place decode complex values
    # [[a,b],[c,d],...] -> [a + ib,c + id,...]
    json_decoder.decode_pulse_qobj(qobj)

    print(datetime.now(), "IN REST API CALLING RUN_EXPERIMENTS")
    connector.run_experiments(
        PulseQobj.from_dict(qobj), enable_traceback=True, job_id=job_id
    )
