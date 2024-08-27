# global variables
from datetime import datetime

from qiskit.providers.ibmq.utils import json_decoder
from qiskit.qobj import PulseQobj

from app.libs.quantum_executor.utils.serialization import iqx_rld

from ...libs.quantum_executor.qiskit.executor import QiskitDynamicsExecutor, QiskitDynamicsPulseSimulator1Q


from ..utils.fixtures import get_fixture_path, load_fixture

connector = QiskitDynamicsPulseSimulator1Q(
    config_file=get_fixture_path("simulator-backend.yml")
)


def test_job_transpile():



    job_dict = load_fixture("h_gate_amplitude_corrected.json")

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
    
