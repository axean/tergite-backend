# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020
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
import settings

# settings
STORAGE_ROOT = settings.STORAGE_ROOT
LABBER_MACHINE_ROOT_URL = settings.LABBER_MACHINE_ROOT_URL

REST_API_MAP = {"scenarios": "/scenarios"}

calibrated = False


async def check_calib_status():
    while 1:
        print("Checking the status of calibration:", end=" ")
        # mimick work
        await asyncio.sleep(1)

        if calibrated:
            print("All OK")
        else:
            print("Calibration required")
            do_calibrate()

        # we check every 5 seconds
        await asyncio.sleep(5)


def helper_dummy_scenario():
    # a dummy Labber scenario mimicking a calibration routine
    job_id = uuid4()
    array_1 = [x for x in range(10)]
    array_2 = [x for x in range(10, 20, 2)]
    scenario = demodulation_scenario(array_1, array_2)
    scenario.tags.tags = [job_id, "calibration"]
    scenario.log_name += str(job_id)

    return scenario


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
    noise_task = asyncio.create_task(noise())
    calib_task = asyncio.create_task(check_calib_status())
    await server_task
    await noise_task
    await calib_task


### run ###
asyncio.run(main())
