# This code is part of Tergite
#
# (C) Axel Andersson (2022)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Refactored by Martin Ahindura (2024)

import copy
import json
from datetime import datetime
from functools import partial
from pathlib import Path
from traceback import format_exc
from typing import Dict, Optional

import numpy as np
import rich
from qiskit.providers.ibmq.utils.json_encoder import IQXJsonEncoder as PulseQobj_encoder
from qiskit.qobj import PulseQobj
from quantify_core.data.handling import create_exp_folder, gen_tuid
from quantify_scheduler.backends.qblox.helpers import generate_port_clock_to_device_map
from quantify_scheduler.backends.qblox_backend import hardware_compile
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.helpers.mock_instruments import MockLocalOscillator
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from tqdm import tqdm
from tqdm.auto import tqdm

from app.libs.quantify.connector.settings import ConnectorSettings
from app.libs.quantify.scheduler.channel import Channel
from app.libs.quantify.scheduler.experiment import Experiment
from app.libs.quantify.scheduler.instruction import Instruction, meas_settings
from app.libs.quantify.simulator import scqt
from app.libs.quantify.simulator.base import BaseSimulator
from app.libs.quantify.utils import general as general_utils
from app.libs.quantify.utils.logger import ExperimentLogger
from app.libs.storage_file import StorageFile

# A map of simulators and their case-insensitive names as referred to in env file
# in the SIMULATOR_TYPE variable
_SIMULATOR_MAP: Dict[str, BaseSimulator] = {"scqt": scqt.Simulator()}


def get_enum_type(type_str: str):
    dummy_enum = ".".join(type_str.split(".")[:-1])
    dummy_enum = import_python_object_from_string(dummy_enum)
    dummy_type = type_str.split(".")[-1]
    return getattr(dummy_enum, dummy_type)


class QuantifyConnector:
    """The connector to the backend using the quantify core library"""

    setup = InstrumentCoordinator("quantify-connector", add_default_generic_icc=False)

    # heap memory, so that instrument drivers do not get garbage collected
    shared_mem = list()

    def __init__(self: "QuantifyConnector", env_file: str = ".env"):
        settings = ConnectorSettings(env_file=Path(env_file).resolve())
        self.HARDWARE_CONFIG = settings.HARDWARE_CONFIG
        self.hardware_map = {
            clock: port
            for (port, clock), instrument in generate_port_clock_to_device_map(
                self.HARDWARE_CONFIG
            ).items()
        }

        self.SIMULATE = settings.SIMULATE
        self.sim = (
            None
            if not self.SIMULATE
            else _SIMULATOR_MAP.get(settings.simulator_type, None)
        )

        for idx, items in enumerate(
            general_utils.find(self.HARDWARE_CONFIG, "instrument_driver")
        ):
            device_name = items[0]
            # instantiate drivers
            driver_str = self.HARDWARE_CONFIG[device_name]["instrument_driver"]
            driver = import_python_object_from_string(driver_str)
            device_ip = self.HARDWARE_CONFIG[device_name].get(
                "instrument_address", None
            )

            print(settings.DEVICES[idx])
            # for devices in settings.DEVICES:
            if settings.DEVICES[idx].split("_")[0] != "DUMMY":
                # this device is a real instrument
                self._wa_device = device = driver(
                    name=device_name, identifier=device_ip
                )

            # this device is a dummy cluster
            elif self.HARDWARE_CONFIG[device_name]["instrument_type"] == "Cluster":
                print("this device is a dummy cluster")
                if not settings.DUMMY_CFG:
                    raise RuntimeError(
                        f"Cluster devices without ip addresses must specify 'DUMMY_CFG in settings"
                    )

                dummy_cfg = {
                    index + 1: "qblox_instruments.ClusterType." + element
                    for index, element in enumerate(settings.DUMMY_CFG)
                }

                device = driver(
                    name=device_name,
                    identifier=None,
                    dummy_cfg={k: get_enum_type(v) for k, v in dummy_cfg.items()},
                )

            # this device is a dummy module
            elif (
                self.HARDWARE_CONFIG[device_name]["instrument_type"] == "Pulsar_QRM"
                or self.HARDWARE_CONFIG[device_name]["instrument_type"] == "Pulsar_QCM"
            ):
                device = driver(
                    name=device_name,
                    dummy_type=get_enum_type(
                        "qblox_instruments.PulsarType."
                        + (self.HARDWARE_CONFIG[device_name]["instrument_type"]).upper()
                    ),
                )
            else:
                # QCoDeS devices
                device = MockLocalOscillator(device_name)

            QuantifyConnector.shared_mem.append(device)
            rich.print(
                f"Instantiated {driver_str.split('.')[-1]} driver for '{device_name}'"
            )

            # add ic component, skip generic component since added automatically
            ic_comp_str = self.HARDWARE_CONFIG[device_name].get(
                "instrument_component", None
            )
            if ic_comp_str:
                ic_component = import_python_object_from_string(ic_comp_str)
                component = ic_component(device)
                QuantifyConnector.setup.add_component(component)
                rich.print(
                    f"Added '{component.name}' to instrument coordinator '{QuantifyConnector.setup.name}'"
                )

            # devices which don't have ic_comp_str are assumed to be QCoDeS devices
            # then the setting dict is just a list of QCoDeS commands to apply
            else:
                for command, value in self.HARDWARE_CONFIG[device_name].items():
                    if hasattr(device, command) and callable(
                        qcodes_command := getattr(device, command)
                    ):
                        qcodes_command(value)
                        rich.print(f"Set '{command}' to {value}")

    def register_job(self: "QuantifyConnector", tag: str = ""):
        # TODO: The fields TUID, FOLDER, and logger could be class properties
        self.TUID = gen_tuid()
        self.FOLDER = Path(create_exp_folder(tuid=self.TUID, name=tag))
        self.logger = ExperimentLogger(self.TUID)

        self.logger.info(f"Registered job: {self.TUID}")
        self.logger.info(
            f"Loaded hardware configuration: {json.dumps(self.HARDWARE_CONFIG, indent=4)}"
        )
        self.logger.info(
            f"Generated hardware map: {json.dumps(self.hardware_map, indent=4)}"
        )

    def reset(self: "QuantifyConnector") -> list:
        # FIXME
        self.wait_external_clock = True  # TODO Workaround
        print("SKIPPING CLUSTER RESET")
        return

        # TODO: Why do we have the cluster reset here?

        # reset all components that can be reset
        QuantifyConnector.setup.stop()
        names = list()
        for component in QuantifyConnector.setup.components():
            try:
                instrument = QuantifyConnector.setup.get_component(component).instrument
                instrument.reset()
                names.append(instrument.name)
            except NotImplementedError:
                pass
            finally:
                # WORKAROUND: we need to wait for external clock refs. every time we reset
                # TODO: wait_external_clock could be property of QuantifyConnector
                self.wait_external_clock = True

            try:
                instrument = QuantifyConnector.setup.get_component(component).instrument
                # WORKAROUND: instrument coordinator doesn't set ref on prepare(), so do it manually
                instrument.reference_source(
                    str.lower(self.HARDWARE_CONFIG[instrument.name]["ref"])
                )
            except NotImplementedError:
                pass

        # standardize attenuation
        if hasattr(self, "_wa_device"):
            # TODO: Here, we can use the new @timer decorator from the benchmarking package
            print(datetime.now(), "STARTING FIX ATT")
            self._fix_attenuation()
            print(datetime.now(), "FINISHED FIX ATT")

        # return names of instruments that were reset
        return names

    def _fix_attenuation(self: "QuantifyConnector"):
        # FIXME: Hardcoded
        # breakpoint()

        self._wa_device.module16.out0_att(0)  # readout module
        self._wa_device.module16.in0_att(0)  # readout module

        self._wa_device.module1.out0_att(12)  # qubit 1 drive module
        self._wa_device.module2.out0_att(12)  # qubit 2 drive module
        # self._wa_device.module3.out0_att(22)  # qubit 3 drive module
        # self._wa_device.module4.out0_att(22)  # qubit 4 drive module
        # self._wa_device.module5.out0_att(22)  # qubit 5 drive module

        rich.print("Set all QCM-RF output attenuation to 12 dB")
        rich.print("Set QRM-RF input attenuation to 0 dB")
        rich.print("Set QRM-RF output attenuation to 0 dB")

    def run(self: "QuantifyConnector", experiment: Experiment, /):
        QuantifyConnector.setup.stop()

        # compile to hardware
        # TODO: Here, we can use the new @timer decorator in the benchmarking package
        t1 = datetime.now()
        if self.SIMULATE:
            compiled_schedule = hardware_compile(
                schedule=experiment.schedule, hardware_cfg=self.HARDWARE_CONFIG
            )
        else:
            absolute_timed_schedule = determine_absolute_timing(
                copy.deepcopy(experiment.schedule)
            )
            compiled_schedule = hardware_compile(
                schedule=absolute_timed_schedule, hardware_cfg=self.HARDWARE_CONFIG
            )

        t2 = datetime.now()
        print(t2 - t1, "DURATION OF COMPILING")

        # log the sequencer assembler programs and the schedule timing table
        self.logger.log_Q1ASM_programs(compiled_schedule)
        self.logger.log_schedule(compiled_schedule)

        # upload schedule to instruments & arm sequencers
        QuantifyConnector.setup.prepare(compiled_schedule)

        if self.wait_external_clock:
            # this avoids the Missing clock reference error
            print("IF THIS PRINTS, A SLEEP IS ACTIVATED")
            # sleep(0.1)
            self.wait_external_clock = False

        # start experiment
        # TODO: Here, we can use the new @timer decorator from the benchmarking package
        t3 = datetime.now()
        QuantifyConnector.setup.start()

        # wait for program to finish and return acquisition
        # TODO: What is the return type of retrieve_acquisition()?
        results = QuantifyConnector.setup.retrieve_acquisition()
        print(f"{results=}")
        t4 = datetime.now()
        print(t4 - t3, "DURATION OF MEASURING")
        return results

    def simulate(self: object, experiment: Experiment, /):
        schedule = experiment.schedule
        compiled_sched = self.sim.compile(schedule)

        self.logger.log_schedule(compiled_sched)

        return self.sim.run(compiled_sched, output="voltage_single_shot")

    def construct_experiments(self: "QuantifyConnector", qobj: PulseQobj, /) -> list:
        # storage array
        tx = list()

        for experiment_index, experiment in enumerate(qobj.experiments):
            instructions = map(
                partial(
                    Instruction.from_qobj,
                    config=qobj.config,
                    hardware_map=self.hardware_map,
                ),
                experiment.instructions,
            )
            instructions = [item for sublist in instructions for item in sublist]

            # create a nice name for the experiment.
            experiment.header.name = StorageFile.sanitized_name(
                experiment.header.name, experiment_index + 1
            )

            # convert OpenPulse experiment to Quantify schedule
            tx.append(
                Experiment(
                    header=experiment.header,
                    instructions=instructions,
                    config=qobj.config,
                    channels=frozenset(
                        Channel(
                            clock=i.channel,
                            frequency=0.0,
                        )
                        for i in instructions
                    ),
                    logger=self.logger,
                )
            )

        self.logger.info(f"Translated {len(tx)} OpenPulse experiments.")
        return tx

    def debug_save_qobj(self: "QuantifyConnector", qobj: PulseQobj):
        """Saves the incoming PulseQobj for debugging.
        This is a re-encoding when using the rest_api, but it is needed for local debugging.
        TODO: Avoid re-encoding for external jobs.
        """
        f = self.FOLDER / "qobj.json"
        with open(f, mode="w") as qj:
            json.dump(qobj.to_dict(), qj, cls=PulseQobj_encoder, indent="\t")
        rich.print(f"Saved PulseQobj at {f}")
        self.logger.info(f"Saved PulseQobj at {f}")

    def run_experiments(
        self: "QuantifyConnector",
        qobj: PulseQobj,
        /,
        *,
        enable_traceback: bool = True,
        job_id: str = None,
    ) -> Optional[Path]:
        """Runs the experiments and returns the results file path

        Args:
            qobj: the Quantum object that is to be executed
            enable_traceback: whether to show the traceback of errors or not
            job_id: the ID of the job

        Returns:
            the path to the results obtained after measurement
        """
        # breakpoint()
        self.debug_save_qobj(qobj)
        results_file_path: Optional[Path] = None
        try:
            self.logger.info(f"Reset instruments: {self.reset()}")

            # unwrap pulse library
            qobj.config.pulse_library = {
                i.name: np.asarray(i.samples) for i in qobj.config.pulse_library
            }

            # translate qobj experiments to quantify schedules
            # TODO: Sometimes, we have still print statements, can we replace them with loggers?
            print(datetime.now(), "IN RUN_EXPERIMENTS, START CONSTRUCTING")
            tx = self.construct_experiments(qobj)

            program_settings = meas_settings(qobj.config)
            for k, v in program_settings.items():
                self.logger.info(f"Set {k} to {v}")

            # create a storage hdf file
            filename = "measurement.hdf5" if job_id is None else f"{job_id}.hdf5"
            results_file_path = self.FOLDER / filename
            storage = StorageFile(
                results_file_path,
                mode="w",
                job_id=job_id,
                tuid=self.TUID,
                meas_return=program_settings["meas_return"],
                meas_return_cols=program_settings["meas_return_cols"],
                meas_level=program_settings["meas_level"],
                memory_slot_size=qobj.config.memory_slot_size,
            )

            # store numpy header metadata
            storage.store_qobj_header(qobj_header=qobj.header.to_dict())

            # run all experiments and store acquisition data
            for experiment_index, experiment in enumerate(
                tqdm(
                    tx,
                    ascii=" #",
                    desc=self.TUID,
                )
            ):
                print(datetime.now(), "IN RUN_EXPERIMENTS, START RUN")

                if self.SIMULATE:
                    experiment_data = self.simulate(experiment)
                else:
                    experiment_data = self.run(experiment)

                experiment_data = experiment_data.to_dict()

                storage.store_experiment_data(
                    experiment_data=experiment_data,
                    name=experiment.header.name,
                )
                storage.store_graph(graph=experiment.dag, name=experiment.header.name)

            self.logger.info(f"Stored measurement data at {storage.file.filename}")

            rich.print(
                ok_str := f"Completed {job_id if job_id else 'local job'} with tuid {self.TUID}."
            )
            self.logger.info(ok_str)

        # record exceptions
        except Exception as e:
            exc_str = f"\n{format_exc()}"
            if enable_traceback:
                rich.print(exc_str)
            self.logger.error(exc_str)

            rich.print(
                fail_str := f"Failed {job_id if job_id else 'local job'} with tuid {self.TUID}. Error: {repr(e)}"
            )
            self.logger.info(fail_str)
            raise e

        # cleanup, regardless if job failed or succeeded
        finally:
            try:
                storage.file.close()
            except UnboundLocalError:
                pass  # no storage to close
            self.reset()

        return results_file_path
