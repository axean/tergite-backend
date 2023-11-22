# This code is part of Tergite
# (C) Copyright Chris Warren 2021
# (C) Copyright Abdullah Al Amin 2022
# (C) Copyright David Wahlstedt 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Modified:
#
# - Martin Ahindura 2023

import numpy as np

import Labber


def parse_labber_logfile(file: Labber.LogFile):
    """
    Parse a Labber LogFile to return a dictionary with only the swept  Step
    Channels as well as the Log Channels formatted to the proper shape

    Args:
        file: Labber logfile object corresponding to the Labber hdf5 file

    Returns:
        xdict (Dict):
            A formatted dictionary containing the data associated with each
            channel that was swept if any.

        ydict (Dict):
            A formatted dictionary containing the data associated with each
            log channel with its values sorted to the shape of what was swept
    """
    xdict = _get_step_data(file)
    ydict = _get_log_data(file)
    xshape = []
    # Labber records step channels top down, but stores y data from the bottom
    # up so the reverse is needed
    for key, d in xdict.items():
        xlen = len(d["values"])
        xshape.append(xlen)
    xshape.reverse()
    # reshape y data based on whether the data is a vector or not
    for key in ydict.keys():
        # If the output y-data is a vector the last index is the number of
        # entries in that vector
        if ydict[key]["meta"]["vector"]:
            ydict[key]["values"] = np.reshape(ydict[key]["values"], [*xshape, -1])
        # If the data is just a single x-sweep or single point it is formated
        elif (len(xshape) == 1) or (len(xshape) == 0):
            pass
        # If the data is a higher order sweep it needs to format
        else:
            ydict[key]["values"] = np.reshape(ydict[key]["values"], xshape)

    return xdict, ydict


def get_meta_data(file, name_list):
    log = Labber.LogFile(file)
    chan_dict = log.getChannelValuesAsDict(True)
    meta_dict = {}
    for name in name_list:
        temp_lst = list(filter(lambda x: name in x, chan_dict.keys()))
        # If there is only one entry flatten it to the top level dictionary
        if len(temp_lst) == 1:
            meta_dict[name] = chan_dict[temp_lst[0]]
        # Otherwise not enough information at this stage to make a determination
        # just return all entries and let the higher level decide the format
        else:
            meta_dict[name] = {key: chan_dict[key] for key in temp_lst}
    return meta_dict


def _get_step_data(log):
    """
    Wrapper around a Labber LogFile to extract out any channel sweep and
    return those channels

    args:
        log (Labber LogFile):
            Input logfile you want to parse

    returns:
        d (Dict):
            A dictionary containing the entries of the Step Channels that
            were swept
    """
    step_data = log.getStepChannels()
    d = {}
    for i, data in enumerate(step_data):
        if len(data["values"]) > 1:
            d["x{}".format(i)] = data
    return d


def _get_log_data(log):
    """
    Wrapper around a Labber LogFile to extract out all log channels of the
    measurement.

    args:
        log (Labber LogFile):
            Input logfile to parse the results from

    returns:
        d (Dict)
            A dictionary containing the entries of the Log Channels
    """
    log_channels = log.getLogChannels()
    d = {}
    for i, chan in enumerate(log_channels):
        d["y{}".format(i)] = {"meta": chan, "values": log.getData(chan["name"])}
    return d
