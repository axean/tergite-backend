# This code is part of Tergite
#
# (C) Johan Blomberg, Gustav Grännsjö 2020
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright David Wahlstedt 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import asyncio
from random import random, randint
from uuid import uuid4
from pathlib import Path
import json
import requests
from enum import Enum
import settings
import datetime
import redis
import calibration_scripts.calibration_mockup as cals
import time
from tempfile import gettempdir


# settings
STORAGE_ROOT = settings.STORAGE_ROOT
LABBER_MACHINE_ROOT_URL = settings.LABBER_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
CALIBRATION_SUPERVISOR_PORT = settings.CALIBRATION_SUPERVISOR_PORT

LOCALHOST = "localhost"

REST_API_MAP = {"jobs": "/jobs"}

# Set up Redis
red = redis.Redis(decode_responses=True)

node_recal_statuses = {}

# Maps names of calibration routines to their corresponding functions
MEASUREMENT_JOBS = {
    "check_res_spect": cals.check_res_spect,
    "cal_res_spect": cals.calibrate_res_spect,
    "check_two-tone": cals.check_two_tone,
    "cal_two-tone": cals.calibrate_two_tone,
    "check_rabi": cals.check_rabi,
    "cal_rabi": cals.calibrate_rabi,
    "check_fidelity": cals.check_fidelity,
    "cal_fidelity": cals.calibrate_fidelity,
}

# Global variable to check the identity of incoming "job done" messages
requested_job_id = ""


async def check_calib_status(job_done):
    global already_verified

    while 1:
        print("Checking the status of calibration:", end=" ")

        # mimick work
        time.sleep(1)

        print("\n------ STARTING MAINTAIN -------\n")
        await maintain_all(job_done)
        print("\n------ MAINTAINED -------\n")

        # Wait a while between checks
        await asyncio.sleep(15)


async def maintain_all(job_done):
    # Get the topological order of DAG nodes
    topo_order = red.lrange("topo_order", 0, -1)
    global node_recal_statuses
    node_recal_statuses = {}

    for node in topo_order:
        node_recal_statuses[node] = await maintain(node, job_done)


async def maintain(node, job_done):
    print(f"Maintaining node {node}")

    state_ok = check_state(node)

    # If state is fine, then no further maintenance is needed.
    if state_ok:
        print(f"Check_state returned true for node {node}. No calibration.")
        return False

    # Perform check_data
    status = await check_data(node, job_done)
    if status == DataStatus.in_spec:
        print(f"Check_data returned in_spec for node {node}. No calibration.")
        return False

    if status == DataStatus.bad_data:
        print(f"Check_data returned bad_data for node {node}. Diagnosing dependencies.")
        deps = red.lrange(f"m_deps:{node}", 0, -1)
        await diagnose_loop(deps, job_done)
    # status is out of spec: no need to diagnose, go directly to calibration
    print(f"Calibration necessary for node {node}. Calibrating...")
    await calibrate(node, job_done)
    return True


def check_state(node):
    # Find dependencies
    deps = red.lrange(f"m_deps:{node}", 0, -1)

    for dep in deps:
        dep_recalibrated = node_recal_statuses[dep]
        if dep_recalibrated:
            print(
                f"Node {dep} needed a recalibration, so check_state for {node} failed"
            )
            return False

    params = red.lrange(f"m_params:{node}", 0, -1)
    for param in params:
        data = red.hget(f"param:{param}", "value")
        if data is None:
            print(f"Parameter {param} didn't exist, so check_state for {node} failed")
            return False
    return True


class DataStatus(Enum):
    in_spec = 1
    out_of_spec = 2
    bad_data = 3


async def check_data(node, job_done):
    # Run the node's associated measurement to check the node's data
    print("")
    mk_measurement_job = MEASUREMENT_JOBS[red.hget(f"measurement:{node}", "check_fn")]
    job = mk_measurement_job()
    job_id = job["job_id"]
    print(f"Requesting check job with {job_id=} for {node=} ...")
    await request_job(job, job_done)
    # Wait for data logistics TODO Change into waiting for an ack from post-processing

    params = red.lrange(f"m_params:{node}", 0, -1)
    for param in params:
        # Fetch the values we got from the measurement's postprocessing
        # TODO: Read from the actual param name, not 'job_id'
        result_key = "results:job_id"
        result = red.get(result_key)
        print(
            f"For {param=}, from Redis we read {result_key} from postprocessing: {result}"
        )
        if result == None:
            print(f"Warning: no entry found for key {result_key}")
        # TODO ensure value is within thresholds

    # TODO return status based on the above param checks instead of deciding at random
    num = random()
    if num < 0.8:
        print(f"Check_data for {node} gives IN_SPEC")
        return DataStatus.in_spec
    if num < 0.95:
        print(f"Check_data for {node} gives OUT_OF_SPEC")
        return DataStatus.out_of_spec
    print(f"Check_data for {node} gives BAD_DATA")
    return DataStatus.bad_data


async def diagnose_loop(initial_nodes, job_done):
    print(f"Starting diagnose for nodes {initial_nodes}")
    # To avoid recursion, we use this while loop function to perform a depth-first diagnose.
    nodes_to_diag = initial_nodes
    nodes_to_measure = []
    diagnosed_nodes = []
    # While there are still nodes to diagnose, keep going
    while nodes_to_diag:
        print(f"Nodes left to diag: {initial_nodes}")
        # Make sure we don't revisit nodes.
        while nodes_to_diag[0] in diagnosed_nodes:
            nodes_to_diag.pop(0)
        # Diagnose the current node, to check if its dependencies needs to be
        # diagnosed, and if this node has to be recalibrated.
        to_diag, to_measure = await diagnose(nodes_to_diag[0], job_done)
        print(f"To_diag = {to_diag}")
        # Mark that we've diagnosed this node
        diagnosed_nodes.append(nodes_to_diag[0])
        # Add the newest set of dependencies first, and remove the current node from the list to diagnose. (Depth first)
        to_diag.extend(nodes_to_diag[1:])
        nodes_to_diag = to_diag

        nodes_to_measure.extend(to_measure)
        print(f"To_diag = {nodes_to_diag}, to_measure = {nodes_to_measure}")

    # Sort nodes to measure in topological order
    topo_order = red.lrange("topo_order", 0, -1)
    nodes_to_measure = [x for x in topo_order for y in nodes_to_measure if x == y]

    for node in nodes_to_measure:
        print(f"(Diag) measuring {node}")
        await calibrate(node, job_done)


async def diagnose(node, job_done):
    status = await check_data(node, job_done)
    if status == DataStatus.in_spec:
        # In spec, no further diag needed, and no recalibration
        return [], []
    if status == DataStatus.out_of_spec:
        # Out of spec, recalibrate, but no deeper diag needed
        return [], [node]
    if status == DataStatus.bad_data:
        # Noise data, diagnose deeper down, and recalibrate
        deps = red.lrange(f"m_deps:{node}", 0, -1)
        return deps, [node]


async def calibrate(node, job_done):
    print("")
    # TODO Add features
    params = red.lrange(f"m_params:{node}", 0, -1)

    # Run the node's associated measurement to check the node's data
    mk_measurement_job = MEASUREMENT_JOBS[
        red.hget(f"measurement:{node}", "calibration_fn")
    ]
    job = mk_measurement_job()
    job_id = job["job_id"]
    print(f"Requesting calibration job with {job_id=} for {node=} ...")
    await request_job(job, job_done)
    # Wait for data logistics TODO Change into waiting for an ack from post-processing

    print("")

    for param in params:
        # Fetch unit and parameter lifetime
        unit = red.hget(f"m_params:{node}:{param}", "unit")
        lifetime = red.hget(f"m_params:{node}:{param}", "timeout")

        # Fetch the values we got from the calibration's postprocessing
        # TODO: Read from the actual param name, not 'job_id'
        result_key = "results:job_id"
        result = red.get(result_key)
        print(
            f"For {param=}, from Redis we read {result_key} from postprocessing: {result}"
        )
        if result == None:
            print(f"Warning: no entry found for key {result_key}")
            result = "not found"  # should investigate why this happens

        red.hset(f"param:{param}", "name", param)
        red.hset(
            f"param:{param}",
            "date",
            datetime.datetime.now().replace(microsecond=0).isoformat() + "Z",
        )
        red.hset(f"param:{param}", "unit", unit)
        red.hset(f"param:{param}", "value", result)

        # Set expiry date
        # TODO replace with flagging system to mark outdated nodes
        red.expire(f"param:{param}", lifetime)


async def request_job(job, job_done):
    global requested_job_id

    job_id = job["job_id"]
    requested_job_id = job_id

    tmpdir = gettempdir()
    file = Path(tmpdir) / str(uuid4())
    with file.open("w") as dest:
        json.dump(job, dest)

    with file.open("r") as src:
        files = {"upload_file": src}
        url = str(BCC_MACHINE_ROOT_URL) + REST_API_MAP["jobs"]
        response = requests.post(url, files=files)

        # Right now the Labber Connector sends a response *after*
        # executing the scenario i.e., the POST request is *blocking*
        # until after the measurement execution this will change in
        # the future; it should just ack a succesful upload of a
        # scenario and nothing more

        if response:
            file.unlink()
            print("Job has been successfully sent")
        else:
            print("request_job failed")

    await job_done.wait()
    job_done.clear()

    print("")


async def handle_message(reader, writer, job_done):
    global requested_job_id

    addr = writer.get_extra_info("peername")
    data = await reader.read(100)
    message = data.decode()

    try:
        [kind, body] = message.split(":")
        if kind == "job_done":
            job_id = body
            if job_id == requested_job_id:
                print(f'handle_message: Received "job done", {job_id=!r} from {addr!r}')
                # notify request_job to proceed
                job_done.set()
            else:
                print(
                    f'handle_message: Received *unexpected*  "job done" message with \
                    {job_id=!r} from {addr!r}'
                )
        else:
            print(f"handle_message: Unknown message kind: {kind}, {message=}")
    except Exception:
        print(f"handle_message: Couldn't parse incoming message: {message}")

    writer.close()


async def message_server(job_done):
    server = await asyncio.start_server(
        lambda reader, writer: handle_message(reader, writer, job_done),
        LOCALHOST,
        CALIBRATION_SUPERVISOR_PORT,
    )
    async with server:
        await server.serve_forever()


async def main():
    # To wait for messages from postprocessing
    job_done = asyncio.Event()

    server_task = asyncio.create_task(message_server(job_done))
    calib_task = asyncio.create_task(check_calib_status(job_done))

    await server_task
    await calib_task


### run ###
asyncio.run(main())
