# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek, Andreas Bengtsson 2020,
# (C) Copyright Abdullah-Al Amin 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import json
import numpy as np
from pathlib import Path
from tempfile import gettempdir
import toml

from Labber import Scenario
from Labber import ScriptTools


def demodulation_scenario(signal_array, demod_array):
    # create and add instruments
    s = Scenario()
    instr_signal = s.add_instrument("Simple Signal Generator", name="Sine")
    instr_demod = s.add_instrument("Signal Demodulation", name="Demod")

    # set a few instrument settings
    instr_demod.values["Use phase reference signal"] = False
    instr_demod.values["Length"] = 1.0

    # add signal connections between channels
    s.add_connection("Sine - Signal", "Demod - Input data")

    # add step items, values can be defined with np array or keywords
    s.add_step("Sine - Frequency", signal_array)
    s.add_step("Demod - Modulation frequency", demod_array)

    # add log channels
    s.add_log("Demod - Value")

    # set metadata
    s.comment = "Comment for log"
    s.tags.project = "My project"
    s.tags.user = "John Doe"
    s.tags.tags = ["Tag 1", "Tag 2/Subtag"]

    # set timing info
    s.wait_between = 0.01

    return s


def qobj_scenario(job):
    supported_gates = set(
        [
            "u1",
            "u2",
            "u3",
            "rx",
            "ry",
            "rz",
            "x",
            "y",
            "z",
            "h",
            "cz",
            "cx",
            "measure",
            "s",
            "sdg",
            "barrier",
        ]
    )
    scenario_template_filepath = Path("./qiskit_qasm_scenario_template.json")
    calibration_filepath = Path("./qiskit_qasm_calibration_config.json")

    qobj = job["params"]["qobj"]

    def validate_gate(gate):
        if gate["name"] not in supported_gates:
            raise ValueError("{} is not supported by the hardware".format(gate["name"]))

        if "conditional" in gate:
            raise ValueError("Conditional gates are not supported by the hardware")

    def validate_job(qobj):
        if qobj["type"] != "QASM":
            raise ValueError("Only QASM-type jobs are supported.")

    validate_job(qobj)
    for ins in qobj["experiments"][0]["instructions"]:
        validate_gate(ins)

    s = Scenario(scenario_template_filepath)

    mqpg = s.get_instrument(name="Pulses")
    n_qubits = qobj["config"].get("n_qubits", 1)
    mqpg.values["Sequence"] = "QObj"
    mqpg.values["QObj JSON"] = json.dumps(qobj["experiments"])

    # configure number of shots
    update_step_single_value(s, "QA - Samples", qobj["config"].get("shots", 1024))

    # update with latest calibration data, if it exists
    if calibration_filepath.exists():
        with open(calibration_filepath, "r") as f:
            calibration = json.load(f)
        update_calibration_data(s, calibration)

    # Configure mulitple experiments
    if len(qobj["experiments"]) == 1:
        s.add_step("Pulses - QObj Iterator", single=0)
    else:
        s.add_step("Pulses - QObj Iterator", np.arange(len(qobj["experiments"])))

    # add relevant log channels
    extraction = job.get("hdf5_log_extraction", None)
    if extraction:
        if extraction.get("waveforms", False):
            add_waveforms(s, n_qubits)
        if extraction.get("voltages", False):
            add_readout_voltages(s, n_qubits)

    # set metadata
    s.log_name = "Test qobj"
    s.comment = "Comment for log"
    s.tags.project = "My project"
    s.tags.user = "Chalmers default user"
    s.tags.tags = ["Qobj"]

    # set timing info
    s.wait_between = 0.2

    return s


## A template scenario file is modified according to the parameters
## that have been passed through job object in order to create a simple
## frequency sweep scenario
def resonator_spectroscopy_scenario(job):
    VNA = "VNA" #"ZNB20"  # "RS"  # 'Keysight'  'Ceyear'

    scenario_template_filepath = Path("./scenario_templates/resonator_spectroscopy_scenario_template_keysight_vna.json")

    # loading Scenario as dictionary
    s_dict = ScriptTools.load_scenario_as_dict(scenario_template_filepath)

    s_prms = job["params"]

    # Updating Step parameters in Scenario dictionary
    for i, stepchannel in enumerate(s_dict["step_channels"]):
        # update: VNA - Output power
        if stepchannel["channel_name"] == VNA + " - Output power":
            if len(s_prms["power"]) == 1: # only single value power is required for the measurement
                s_dict["step_channels"][i]["step_items"][0]["range_type"] = "Single"
                s_dict["step_channels"][i]["step_items"][0]["single"] = s_prms["power"][0]
            elif len(s_prms["power"]) == 3: # multiple step value for power is required for the measurement
                s_dict["step_channels"][i]["step_items"][0]["range_type"] = "Start - Stop"
                s_dict["step_channels"][i]["step_items"][0]["step_type"] ="Fixed # of pts"
                s_dict["step_channels"][i]["step_items"][0]["start"] = s_prms["power"][0]
                s_dict["step_channels"][i]["step_items"][0]["stop"] = s_prms["power"][1]
                s_dict["step_channels"][i]["step_items"][0]["n_pts"] = s_prms["power"][2]
            else:
                raise ValueError("Input Power parameter is not well defined.")
        # update VNA - IF bandwidth
        elif stepchannel["channel_name"] == VNA + " - IF bandwidth":
            s_dict["step_channels"][i]["step_items"][0]["single"] = s_prms["if_bw"]
        # update VNA - Number of averages
        elif stepchannel["channel_name"] == VNA + " - # of averages":
            s_dict["step_channels"][i]["step_items"][0]["single"] = s_prms["num_ave"]
        # update VNA - Start of Sweping frequency
        elif stepchannel["channel_name"] == VNA + " - Start frequency":
            s_dict["step_channels"][i]["step_items"][0]["single"] = s_prms["f_start"]
        # update VNA - Stop of Sweping frequency
        elif stepchannel["channel_name"] == VNA + " - Stop frequency":
            s_dict["step_channels"][i]["step_items"][0]["single"] = s_prms["f_stop"]
        # update VNA - Number of measurement points (data points)
        elif stepchannel["channel_name"] == VNA + " - # of points":
            s_dict["step_channels"][i]["step_items"][0]["single"] = s_prms["num_pts"]

    # Saving Scenario in a temporary file as JSON format
    temp_dir = gettempdir()
    ScriptTools.save_scenario_as_json(s_dict, temp_dir + "/tmp.json")

    # Loading Scenario as object
    s = Scenario(temp_dir + "/tmp.json")

    # set metadata
    s.log_name = "Resonator Spectroscopy"
    s.comment = "Comment for log"
    s.tags.project = "Automatic Calibration project"
    s.tags.user = "Chalmers default user"
    s.tags.tags = ["ResonatorSpectroscopy"]

    # set timing info
    s.wait_between = 0.2

    return s

# A generic scenario creation routine for pulsed resonator
# spectroscopy, two-tone, Rabi, and Ramey calibration steps, using ZI
# and Labber. It can be used to create related measurement scenarios
# as well.
def generic_calib_zi_scenario(job):

    if job["name"] == "ramsey_qubit_freq_correction":
        scenario_template_filepath = Path("./scenario_templates/ramsey_using_general_calib_template.json")
    elif job["name"] == "rabi_qubit_pi_pulse_estimation":
        scenario_template_filepath = Path("./scenario_templates/rabi_using_general_calib_template.json")
    elif job["name"] == "pulsed_two_tone_qubit_spectroscopy":
        scenario_template_filepath = Path("./scenario_templates/pulsed_qubit_spectroscopy_using_general_calib_template.json")
    elif job["name"] == "pulsed_resonator_spectroscopy":
        scenario_template_filepath = Path("./scenario_templates/pulsed_spectroscopy_scenario_template.json")

    # loading Scenario as dictionary
    s_dict = ScriptTools.load_scenario_as_dict(scenario_template_filepath)

    s_prms = job["params"]

    # Updating Step parameters in Scenario dictionary
    for i, stepchannel in enumerate(s_dict["step_channels"]):
        step_channel_i = s_dict["step_channels"][i]["step_items"][0]
        # For Qubit Control RF source settings:
        #if stepchannel["channel_name"] == "Trace Time":
        #    step_channel_i["single"] = s_prms["trace_time"]
        if stepchannel["channel_name"] == "MQPG Control - Sample rate":
            step_channel_i["single"] = s_prms["mqpg_smpl_rate"]
        elif stepchannel["channel_name"] == "MQPG Control - Frequency #1":
            step_channel_i["range_type"] = s_prms["control_freq_range_type"]
            if s_prms["control_freq_range_type"] == "Start - Stop":
                step_channel_i["start"] = s_prms["control_start_freq"]
                step_channel_i["stop"] = s_prms["control_stop_freq"]
                step_channel_i["n_pts"] = s_prms["num_pts"]
            elif s_prms["control_freq_range_type"] == "Single":
                step_channel_i["single"] = s_prms["control_freq"]
        elif stepchannel["channel_name"] == "MQPG Control - Amplitude #1":
            step_channel_i["range_type"] = s_prms["control_amp_range_type"]
            if s_prms["control_amp_range_type"] == "Start - Stop":
                step_channel_i["start"] = s_prms["control_amp_start"]
                step_channel_i["stop"] = s_prms["control_amp_stop"]
                step_channel_i["n_pts"] = s_prms["num_pts"]
            elif s_prms["control_amp_range_type"] == "Single":
                step_channel_i["single"] = s_prms["control_amp"]
        elif stepchannel["channel_name"] == "MQPG Control - Pulse spacing":
            step_channel_i["range_type"] = s_prms["control_pulse_spacing_range_type"]
            if s_prms["control_amp_range_type"] == "Start - Stop":
                step_channel_i["start"] = s_prms["control_pulse_spacing_start"]
                step_channel_i["stop"] = s_prms["control_pulse_spacing_stop"]
                step_channel_i["n_pts"] = s_prms["num_pts"]
            elif s_prms["control_amp_range_type"] == "Single":
                step_channel_i["single"] = s_prms["control_pulse_spacing"]
        elif stepchannel["channel_name"] == "Qubit 2B - Output":
            step_channel_i["single"] = s_prms["control_output_enabled"]
        elif stepchannel["channel_name"] == "Qubit 2B - Power":
            if s_prms["control_power_range_type"] == "Start - Stop":
                step_channel_i["start"] = s_prms["control_start_power"]
                step_channel_i["stop"] = s_prms["control_stop_power"]
                step_channel_i["n_pts"] = s_prms["num_pts"]
            elif s_prms["control_power_range_type"] == "Single":
                step_channel_i["single"] = s_prms["control_power"]
        # For Readout RF source frequency and power settings:
        elif stepchannel["channel_name"] == "QA_Carrier - Frequency":
            step_channel_i["range_type"] = s_prms["readout_freq_range_type"]
            if s_prms["readout_freq_range_type"] == "Start - Stop":
                step_channel_i["start"] = s_prms["readout_start_freq"]
                step_channel_i["stop"] = s_prms["readout_stop_freq"]
                step_channel_i["n_pts"] = s_prms["num_pts"]
            elif s_prms["readout_freq_range_type"] == "Single":
                step_channel_i["single"] = s_prms["readout_resonance_freq"]
        elif stepchannel["channel_name"] == "QA_Carrier - Power":
            step_channel_i["range_type"] = s_prms["readout_power_range_type"]
            if s_prms["readout_power_range_type"] == "Start - Stop":
                step_channel_i["start"] = s_prms["readout_power_start"]
                step_channel_i["stop"] = s_prms["readout_power_stop"]
                step_channel_i["n_pts"] = s_prms["num_pts"]
            elif s_prms["readout_power_range_type"] == "Single":
                step_channel_i["single"] = s_prms["readout_power"]
        # Readout settings for MQPG
        elif stepchannel["channel_name"] == "MQPG Readout - Readout frequency #1":
            step_channel_i["single"] = s_prms["readout_freq"]
        elif stepchannel["channel_name"] == "MQPG Readout - Readout amplitude #1":
            step_channel_i["range_type"] = s_prms["readout_amp_range_type"]
            if s_prms["readout_amp_range_type"] == "Start - Stop":
                step_channel_i["start"] = s_prms["readout_amp_start"]
                step_channel_i["stop"] = s_prms["readout_amp_stop"]
                step_channel_i["n_pts"] = s_prms["num_pts_other_axis"]
            elif s_prms["readout_power_range_type"] == "Single":
                step_channel_i["single"] = s_prms["readout_amp"]
        elif stepchannel["channel_name"] == "MQPG Readout - Readout duration":
            step_channel_i["single"] = s_prms["readout_duration"]
        # HDAWG Marker Pulse Setting, Marker is used as trigger for QA
        elif stepchannel["channel_name"] == "HDAWG - Internal trigger period":
            step_channel_i["range_type"] = s_prms["hdawg_trigger_range_type"]
            if s_prms["hdawg_trigger_range_type"] == "Start - Stop":
                step_channel_i["start"] = s_prms["hdawg_trigger_period_start"]
                step_channel_i["stop"] = s_prms["hdawg_trigger_period_stop"]
                step_channel_i["n_pts"] = s_prms["num_pts"]
            elif s_prms["hdawg_trigger_range_type"] == "Single":
                step_channel_i["single"] = s_prms["hdawg_int_trig_period"]
        elif stepchannel["channel_name"] == "HDAWG - Output 1 Marker 1 duration":
            step_channel_i["single"] = s_prms["hdwag_marker_duration"]
        # UHFQA Settings
        elif stepchannel["channel_name"] == "QA_DEV2346 - Integration Length":
            step_channel_i["single"] = s_prms["qa_integration_length"]
        elif stepchannel["channel_name"] == "QA_DEV2346 - Delay":
            step_channel_i["single"] = s_prms["qa_delay"]
        elif stepchannel["channel_name"] == "QA_DEV2346 - Averages":
            step_channel_i["single"] = s_prms["qa_avg"]


    # Saving Scenario in a temporary file as JSON format
    temp_dir = gettempdir()
    ScriptTools.save_scenario_as_json(s_dict, temp_dir + "/tmp.json")

    # Loading Scenario as object
    s = Scenario(temp_dir + "/tmp.json")

    # set metadata
    # these needs to be automated by redis entries
    s.log_name = job["name"]
    s.comment = "Comment for log"
    s.tags.project = "Automatic Calibration project"
    s.tags.user = "Chalmers default user"
    s.tags.tags = [job["name"]]

    # set timing info
    s.wait_between = 0.2

    return s


def qobj_dummy_scenario(job):
    scenario_template_filepath = Path("./__stub__qiskit_qasm_scenario_template.json")

    qobj = job["params"]["qobj"]

    def validate_job(qobj):
        if qobj["type"] != "QASM":
            raise ValueError("Only QASM-type jobs are supported.")

    validate_job(qobj)
    s = Scenario(scenario_template_filepath)
    instr = s.get_instrument(name="State Discriminator 2 States")
    items = s.step_items
    selector_item = items[0]
    step = selector_item.range_items[0]

    # Fetch number of experiements
    no_experiments = len(qobj["experiments"])
    # Set up a labber sweep over the experiments (or no sweep, if only 1)
    if no_experiments > 1:
        step.range_type = "Start - Stop"
        step.start = 1
        step.stop = no_experiments
        # Stepsize, should always be 1.
        step.step = 1
    else:
        step.range_type = "Single"
        step.single = 1

    instr.values["QObj JSON"] = json.dumps(qobj)
    instr.values["QObj ID"] = qobj["qobj_id"]

    # set metadata
    s.log_name = "Test qobj"
    s.comment = "Comment for log"
    s.tags.project = "My project"
    s.tags.user = "Chalmers default user"
    s.tags.tags = ["Qobj"]

    # set timing info
    s.wait_between = 0.2

    return s


def update_step_single_value(scenario, name, value):
    scenario.get_step(name).range_items[0].single = value


def translate_parameter_name(id_, calibration_parameter):
    parameters = {
        "qubit_frequency": "Qubit {id} Frequency",
        "pi_amplitude": "Qubit {id} Amplitude",
        "drag_coefficient": "Qubit {id} Alpha",
        "readout_frequency": "Res{id} Frequency",
        "readout_amplitude": "Readout - Readout amplitude #{id}",
    }

    return parameters[calibration_parameter].format(id=str(id_))


def update_calibration_data(scenario, calibration):
    for qubit in calibration["qubits"]:
        id_ = qubit["id"]
        for parameter in qubit:
            if parameter == "id":
                continue
            elif parameter in [
                "readout_0_real_voltage",
                "readout_0_imag_voltage",
                "readout_1_real_voltage",
                "readout_1_imag_voltage",
            ]:
                state = scenario.get_instrument("State Discriminator 2 States")
                if parameter == "readout_0_real_voltage":
                    tmp = state.values["Pointer, QB{}-S0".format(id_)]
                    state.values["Pointer, QB{}-S0".format(id_)] = qubit[
                        parameter
                    ] + 1j * np.imag(tmp)
                elif parameter == "readout_0_imag_voltage":
                    tmp = state.values["Pointer, QB{}-S0".format(id_)]
                    state.values["Pointer, QB{}-S0".format(id_)] = (
                        np.real(tmp) + 1j * qubit[parameter]
                    )
                elif parameter == "readout_1_real_voltage":
                    tmp = state.values["Pointer, QB{}-S1".format(id_)]
                    state.values["Pointer, QB{}-S1".format(id_)] = qubit[
                        parameter
                    ] + 1j * np.imag(tmp)
                elif parameter == "readout_1_imag_voltage":
                    tmp = state.values["Pointer, QB{}-S1".format(id_)]
                    state.values["Pointer, QB{}-S1".format(id_)] = (
                        np.real(tmp) + 1j * qubit[parameter]
                    )
            else:
                update_step_single_value(
                    scenario, translate_parameter_name(id_, parameter), qubit[parameter]
                )


def add_waveforms(scenario, n_qubits):
    channel = "Pulses - Trace - {waveform}{id}"
    # Add waveforms for qubit XY and Z lines
    for i in range(n_qubits):
        for j in ["I", "Q", "ZI"]:
            scenario.add_log(channel.format(waveform=j, id=str(i + 1)))

    # Add readout waveforms
    scenario.add_log("Readout - Trace - Readout I")
    scenario.add_log("Readout - Trace - Readout Q")


def add_readout_voltages(scenario, n_qubits):
    channel = "QA - Result {id}"
    for i in range(n_qubits):
        scenario.add_log(channel.format(id=str(i + 1)))
