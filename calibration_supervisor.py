# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek, Johan Blomberg, Gustav Grännsjö 2020
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
from scenario_scripts import demodulation_scenario
from uuid import uuid4
from pathlib import Path
import requests
from enum import Enum
import settings
import datetime
import redis
import cal_routines.dummy_cals as cals
import cal_routines.dummy_checks as checks
import time

# settings
STORAGE_ROOT = settings.STORAGE_ROOT
LABBER_MACHINE_ROOT_URL = settings.LABBER_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
CALIBRATION_SUPERVISOR_PORT = settings.CALIBRATION_SUPERVISOR_PORT

LOCALHOST = "localhost"


REST_API_MAP = {"scenarios": "/scenarios"}

# Set up redis
red = redis.Redis(decode_responses=True)

calibrated = False

node_recal_statuses = {}

# Maps names of calibration routines to their corresponding functions
MEASUREMENT_ROUTINES = {
    "cal_rabi": cals.cal_dummy,
    "check_rabi": checks.check_dummy,
    "cal_two-tone": cals.cal_dummy,
    "check_two-tone": checks.check_dummy,
    "cal_res_spect": cals.cal_dummy,
    "check_res_spect": checks.check_dummy,
    "do_fidelity_cal": cals.cal_dummy,
    "do_fidelity_check": checks.check_dummy,
}


async def check_calib_status():
    global already_verified

    while 1:
        print("Checking the status of calibration:", end=" ")

        # mimick work
        await asyncio.sleep(1)

        print("------ STARTING MAINTAIN -------")
        await maintain_all()
        print("------ MAINTAINED -------")

        # Wait a while between checks
        await asyncio.sleep(15)


async def maintain_all():
    # Get the topological order of DAG nodes
    topo_order = red.lrange("topo_order", 0, -1)
    global node_recal_statuses
    node_recal_statuses = {}

    for node in topo_order:
        node_recal_statuses[node] = await maintain(node)


async def maintain(node):
    print(f"Maintaining node {node}")

    result = await check_state(node)

    # If state is fine, then no further maintenance is needed.
    if result:
        print(f"check_state returned true for node {node}. No calibration.")
        return False

    # Perform check_data
    result = await check_data(node)

    if result == DataStatus.in_spec:
        print(f"check_data returned in_spec for node {node}. No calibration.")
        return False

    if result == DataStatus.bad_data:
        print(f"check_data returned bad_data for node {node}. diagnosing dependencies.")
        deps = red.lrange(f"m_deps:{node}", 0, -1)
        await diagnose_loop(deps)
    # if out of spec, no need to diagnose, go directly to calibration
    print(f"(Maintain) Calibration necessary for node {node}. Calibrating...")
    await calibrate(node)
    return True


async def check_state(node):
    # Find dependencies
    deps = red.lrange(f"m_deps:{node}", 0, -1)

    for dep in deps:
        dep_recalibrated = node_recal_statuses[dep]
        if dep_recalibrated:
            print(f"{dep} needed a recal, so check_state for {node} failed")
            return False

    params = red.lrange(f"m_params:{node}", 0, -1)
    for param in params:
        data = red.hget(f"param:{param}", "value")
        if data is None:
            print(f"parameter {param} didn't exist, so check_state for {node} failed")
            return False
    return True


class DataStatus(Enum):
    in_spec = 1
    out_of_spec = 2
    bad_data = 3


async def check_data(node):
    # Generate a scenario to check the node's data
    scenario = MEASUREMENT_ROUTINES[red.hget(f"measurement:{node}", "check_f")]()
    # Run the scenario through Labber
    print(f"Running check scenario for {node}...")
    await send_scenario(scenario)
    # Wait for data logistics TODO Change into waiting for an ack from post-processing
    time.sleep(2)

    params = red.lrange(f"m_params:{node}", 0, -1)
    for param in params:
        # Fetch the values we got from the measurement
        # TODO read from the actual param name, not 'shots'
        result = red.get("results:shots")
        print(f"from redis we read shots: {result}")

        # TODO ensure value is within thresholds

    # TODO return status based on the above param checks instead of deciding at random
    num = random()
    if num < 0.8:
        print(f"Check_data for {node} gives IN_SPEC")
        return DataStatus.in_spec
    if num < 0.95:
        print(f"Check_data for {node} gives OUT_OF_SPEC")
        return DataStatus.out_of_spec
    print(f"Check_data for {node} gives BAD")
    return DataStatus.bad_data


async def diagnose_loop(initial_nodes):
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
        to_diag, to_measure = await diagnose(nodes_to_diag[0])
        print(f"to_diag = {to_diag}")
        # Mark that we've diagnosed this node
        diagnosed_nodes.append(nodes_to_diag[0])
        # Add the newest set of dependencies first, and remove the current node from the list to diagnose. (Depth first)
        to_diag.extend(nodes_to_diag[1:])
        nodes_to_diag = to_diag

        nodes_to_measure.extend(to_measure)
        print(f"to_diag = {nodes_to_diag}, to_measure = {nodes_to_measure}")

    # Sort nodes to measure in topological order
    topo_order = red.lrange("topo_order", 0, -1)
    nodes_to_measure = [x for x in topo_order for y in nodes_to_measure if x == y]

    for node in nodes_to_measure:
        print(f"(Diag) measuring {node}")
        await calibrate(node)


async def diagnose(node):
    status = await check_data(node)
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


async def calibrate(node):
    # TODO Add features
    params = red.lrange(f"m_params:{node}", 0, -1)

    # Generate a scenario to calibrate the node
    scenario = MEASUREMENT_ROUTINES[red.hget(f"measurement:{node}", "cal_f")]()
    # Run the scenario through Labber
    print(f"Running calibration scenario for {node}...")
    await send_scenario(scenario)

    # Wait for data logistics TODO Change into waiting for an ack from post-processing
    time.sleep(2)

    for param in params:
        # Fetch unit and parameter lifetime
        unit = red.hget(f"m_params:{node}:{param}", "unit")
        lifetime = red.hget(f"m_params:{node}:{param}", "timeout")

        # Fetch the values we got from the calibration
        # TODO read from the actual param name, not 'shots'
        result = red.get("results:shots")
        print(f"from redis we read shots: {result}")

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


async def send_scenario(scenario):
    job_id = scenario.tags.tags[0]
    scenario_file = Path(STORAGE_ROOT) / (job_id + ".labber")
    scenario.save(scenario_file)
    print(f"Scenario generated at {str(scenario_file)}")

    with scenario_file.open("rb") as source:
        files = {
            "upload_file": (scenario_file.name, source),
            "send_logfile_to": (None, str(BCC_MACHINE_ROOT_URL)),
        }
        url = str(LABBER_MACHINE_ROOT_URL) + REST_API_MAP["scenarios"]
        # to be replaced with aiohttp call
        response = requests.post(url, files=files)
        # simulate async request
        await asyncio.sleep(5)
    # right now the Labber Connector sends a response *after* executing the scenario
    # ie the POST request is *blocking* until after the measurement execution
    # this will change in the future; it should just ack a succesful upload of a scenario and nothing more
    if response:
        # clean up
        scenario_file.unlink()
        print("Scenario executed successfully")
    else:
        print("Scenario execution failed")


# NOTE: This message handling is WIP! The calibration loop does *not* depend on it.
# Current status: The messages are ariving, but very late.
async def handle_message(reader, writer):
    data = await reader.read(100)
    message = data.decode()
    addr = writer.get_extra_info("peername")

    print(f"Received {message!r} from {addr!r}")
    writer.close()


async def message_server():
    server = await asyncio.start_server(
        handle_message, LOCALHOST, CALIBRATION_SUPERVISOR_PORT)
    async with server:
        await server.serve_forever()


async def main():
    server_task = asyncio.create_task(message_server())
    calib_task = asyncio.create_task(check_calib_status())
    await server_task
    await calib_task


### run ###
asyncio.run(main())
