# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek, Johan Blomberg, Gustav Grännsjö 2020
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

# settings
STORAGE_ROOT = settings.STORAGE_ROOT
LABBER_MACHINE_ROOT_URL = settings.LABBER_MACHINE_ROOT_URL

REST_API_MAP = {"scenarios": "/scenarios"}

# Set up redis
red = redis.Redis(host='localhost', port=6379, decode_responses=True)

calibrated = False

node_recal_statuses = {}

# Maps names of calibration routines to their corresponding functions
MEASUREMENT_ROUTINES = {
    'cal_rabi': cals.cal_dummy(),
    'check_rabi': checks.check_dummy(),
    'cal_two-tone': cals.cal_dummy(),
    'check_two-tone': checks.check_dummy(),
    'cal_res_spect': cals.cal_dummy(),
    'check_res_spect': checks.check_dummy(),
    'do_fidelity_cal': cals.cal_dummy(),
    'do_fidelity_check': checks.check_dummy()
}

async def check_calib_status():
    global already_verified
    while 1:
        print("Checking the status of calibration:", end=" ")
        # mimick work
        await asyncio.sleep(1)

        print("------ STARTING MAINTAIN -------")
        maintain_all()
        print("------ MAINTAINED -------")

        # if calibrated:
        #     print("All OK")
        # else:
        #     print("Calibration required")
        #     do_calibrate()

        # Wait a while between checks
        await asyncio.sleep(15)


def helper_dummy_scenario():
    # a dummy Labber scenario mimicking a calibration routine
    job_id = uuid4()
    array_1 = [x for x in range(10)]
    array_2 = [x for x in range(10, 20, 2)]
    scenario = demodulation_scenario(array_1, array_2)
    scenario.tags.tags = [job_id, "calibration"]
    scenario.log_name += str(job_id)

    return scenario


def maintain_all():
    # Get the topological order of DAG nodes
    topo_order = red.lrange('topo_order', 0, -1)
    global node_recal_statuses
    node_recal_statuses = {}

    for node in topo_order:
        node_recal_statuses[node] = maintain(node)


def maintain(node):
    print(f'Maintaining node {node}')

    result = check_state(node)

    # If state is fine, then no further maintenance is needed.
    if result:
        print(f'check_state returned true for node {node}. No calibration.')
        return False

    # Perform check_data
    result = check_data(node)

    if result == DataStatus.in_spec:
        print(f'check_data returned in_spec for node {node}. No calibration.')
        return False

    if result == DataStatus.bad_data:
        print(f'check_data returned bad_data for node {node}. diagnosing dependencies.')
        deps = red.lrange(f'm_deps:{node}', 0, -1)
        diagnose_loop(deps)

    print(f'(Maintain) Calibration necessary for node {node}. Calibrating...')
    calibrate(node)
    return True


def check_state(node):
    # Find dependencies
    deps = red.lrange(f'm_deps:{node}', 0, -1)

    for dep in deps:
        dep_recalibrated = node_recal_statuses[dep]
        if dep_recalibrated:
            print(f'{dep} needed a recal, so check_state for {node} failed')
            return False

    params = red.lrange(f'm_params:{node}', 0, -1)
    for param in params:
        data = red.hget(f'param:{param}', 'value')
        if data is None:
            print(f'{param} didn\'t exist, so check_state for {node} failed')
            return False
    return True


class DataStatus(Enum):
    in_spec = 1
    out_of_spec = 2
    bad_data = 3


def check_data(node):
    # Generate a scenario to check the node's data
    scenario = MEASUREMENT_ROUTINES[node['check_f']]()
    # Run the scenario through Labber
    print(f'Running calibration scenario for {node}...')
    send_scenario(scenario)

    # Wait for data logistics
    await asyncio.sleep(0.1)

    params = red.lrange(f'm_params:{node}', 0, -1)
    for param in params:
        # Fetch the values we got from the measurement
        # TODO read from the actual param name, not 'shots'
        result = red.get('results:shots')

        # TODO ensure value is within thresholds

    # TODO return status based on the above param checks instead of deciding at random
    num = random()
    if num < 0.8:
        print(f'Check_data for {node} gives IN_SPEC')
        return DataStatus.in_spec
    if num < 0.95:
        print(f'Check_data for {node} gives OUT_OF_SPEC')
        return DataStatus.out_of_spec
    print(f'Check_data for {node} gives BAD')
    return DataStatus.bad_data


def diagnose_loop(initial_nodes):
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
        to_diag, to_measure = diagnose(nodes_to_diag[0])
        print(f'to_diag = {to_diag}')
        # Mark that we've diagnosed this node
        diagnosed_nodes.append(nodes_to_diag[0])
        # Add the newest set of dependencies first, and remove the current node from the list to diagnose. (Depth first)
        to_diag.extend(nodes_to_diag[1:])
        nodes_to_diag = to_diag

        nodes_to_measure.extend(to_measure)
        print(f"to_diag = {nodes_to_diag}, to_measure = {nodes_to_measure}")

    # Sort nodes to measure in topological order
    topo_order = red.lrange('topo_order', 0, -1)
    nodes_to_measure = [x for x in topo_order for y in nodes_to_measure if x == y]

    for node in nodes_to_measure:
        print(f"(Diag) measuring {node}")
        calibrate(node)


def diagnose(node):
    status = check_data(node)
    if status == DataStatus.in_spec:
        # In spec, no further diag needed, and no recalibration
        return [], []
    if status == DataStatus.out_of_spec:
        # Out of spec, recalibrate, but no deeper diag needed
        return [], [node]
    if status == DataStatus.bad_data:
        # Noise data, diagnose deeper down, and recalibrate
        deps = red.lrange(f'm_deps:{node}', 0, -1)
        return deps, [node]


def calibrate(node):
    # TODO Add features
    params = red.lrange(f'm_params:{node}', 0, -1)

    # Generate a scenario to calibrate the node
    scenario = MEASUREMENT_ROUTINES[node['cal_f']]()
    # Run the scenario through Labber
    print(f'Running calibration scenario for {node}...')
    send_scenario(scenario)

    # Wait for data logistics
    await asyncio.sleep(0.1)

    for param in params:
        # Fetch unit and parameter lifetime
        unit = red.hget(f'm_params:{node}:{param}', 'unit')
        lifetime = red.hget(f'm_params:{node}:{param}', 'timeout')

        # Fetch the values we got from the calibration
        # TODO read from the actual param name, not 'shots'
        result = red.get('results:shots')

        red.hset(f'param:{param}', 'name',  param)
        red.hset(f'param:{param}', 'date',  datetime.datetime.now().replace(microsecond=0).isoformat() + 'Z')
        red.hset(f'param:{param}', 'unit',  unit)
        red.hset(f'param:{param}', 'value', result)

        # Set expiry date
        # TODO replace with flagging system to mark outdated nodes
        # red.expire(f'param:{param}', lifetime)


def calib_routine_1():
    # routine 1

    send_scenario(helper_dummy_scenario())


def calib_routine_2():
    # routine 2

    send_scenario(helper_dummy_scenario())


def send_scenario(scenario):

    job_id = scenario.tags.tags[0]
    scenario_file = Path(STORAGE_ROOT) / (job_id + ".labber")
    scenario.save(scenario_file)
    print(f"Scenario generated at {str(scenario_file)}")

    with scenario_file.open("rb") as source:
        files = {"upload_file": source}
        url = str(LABBER_MACHINE_ROOT_URL) + REST_API_MAP["scenarios"]
        response = requests.post(url, files=files)

    # right now the Labber Connector sends a response *after* executing the scenario
    # this will change in the future; it should just ack a succesful upload of a scenario and nothing more
    if response:
        # clean up
        scenario_file.unlink()
        print("Scenario executed successfully")
    else:
        print("Scenario execution failed")


def do_calibrate():
    global calibrated

    calib_routine_1()
    calib_routine_2()

    calibrated = True
    print("Calibration finished")


async def noise():
    # here we are simulating noise
    # it just occasionally flips the 'calibrated' boolean
    global calibrated

    while 1:
        await asyncio.sleep(randint(15, 25))
        calibrated = False
        # print("(fyi, there was some noise)")


async def handle_message(reader, writer):
    data = await reader.read(100)
    message = data.decode()
    addr = writer.get_extra_info("peername")

    print(f"Received {message!r} from {addr!r}")
    writer.close()


async def message_server():
    server = await asyncio.start_server(handle_message, "127.0.0.1", 8888)
    async with server:
        await server.serve_forever()


async def main():
    server_task = asyncio.create_task(message_server())
    #noise_task = asyncio.create_task(noise())
    calib_task = asyncio.create_task(check_calib_status())
    await server_task
    #await noise_task
    await calib_task


### run ###
asyncio.run(main())
