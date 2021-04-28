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


import os
from Labber import ScriptTools
import Labber

# Labber.ScriptTools.setExePath("/usr/share/Labber/Program/Measurement")

CONFIG_FILE = "local_echo_conf.hdf5"
LOG_FILE = "./log.hdf5"


def labber_job(value):
    if os.path.exists(LOG_FILE):
        print("Removing previous log file")
        os.remove(LOG_FILE)

    labber_object = ScriptTools.MeasurementObject(CONFIG_FILE, LOG_FILE)

    # labber_object.updateValue('Pulse Generator - Plateau #1', 101)
    labber_object.updateValue("AxlRose - Local echo - Double READ", value)
    labber_object.performMeasurement(False)

    f = Labber.LogFile(LOG_FILE)
    print("Number of entries:", f.getNumberOfEntries())
    print("Log channels:")
    log_channels = f.getLogChannels()
    for channel in log_channels:
        print(" " + channel["name"])

    raw_data = f.getData("AxlRose - Local echo - Double READ")
    results = [x for x in (raw_data[0].tolist())]
    print(len(results))

    print("Labber worker finished")
