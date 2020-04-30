from Labber import Scenario
import numpy as np


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


#    # set log name and save to disk
#    s.log_name = "Test signal demodulation"
#    s.save("demodulation_scenario")
