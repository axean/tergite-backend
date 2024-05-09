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
import dataclasses
import enum
import json
import multiprocessing as mp
import os
from datetime import datetime
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
from traceback import format_exc
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import qblox_instruments
import quantify_core.data.handling as dh
import rich
from qiskit.providers.ibmq.utils.json_encoder import IQXJsonEncoder as PulseQobj_encoder
from qiskit.qobj import PulseQobj
from quantify_core.data.handling import create_exp_folder, gen_tuid
from quantify_scheduler.backends.qblox.helpers import generate_port_clock_to_device_map
from quantify_scheduler.backends.qblox_backend import hardware_compile
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent
from tqdm import tqdm
from tqdm.auto import tqdm

import settings
from app.libs.storage_file import StorageFile

from .scheduler.channel import Channel
from .scheduler.experiment import Experiment
from .scheduler.instruction import Instruction, meas_settings
from .simulator import scqt
from .simulator.base import BaseSimulator
from .utils.config import ClusterModuleType, KernelConfig
from .utils.logger import ExperimentLogger

# A map of simulators and their case-insensitive names as referred to in env file
# in the SIMULATOR_TYPE variable
_SIMULATOR_MAP: Dict[str, BaseSimulator] = {"scqt": scqt.Simulator()}

_QBLOX_CLUSTER_TYPE_MAP: Dict[ClusterModuleType, qblox_instruments.ClusterType] = {
    ClusterModuleType.QCM: qblox_instruments.ClusterType.CLUSTER_QCM,
    ClusterModuleType.QRM: qblox_instruments.ClusterType.CLUSTER_QRM,
    ClusterModuleType.QCM_RF: qblox_instruments.ClusterType.CLUSTER_QCM_RF,
    ClusterModuleType.QRM_RF: qblox_instruments.ClusterType.CLUSTER_QRM_RF,
}


def serve(
    channel: Connection,
    config_file: Union[str, bytes, os.PathLike] = settings.KERNEL_CONFIG_FILE,
):
    """Serves the quantify

    Args:
        channel: The channel on which to communicate with the external world
        config_file: the path to the quantify hardware configuration file
    """
    connector = Kernel(config_file=config_file)

    while True:
        message = channel.recv()

        if isinstance(message, KernelMessage):
            if message.type == KernelMessageType.CLOSE:
                # connector.close()
                break

            if message.type == KernelMessageType.RUN:
                qobj = message.payload.pop("qobj")
                results_file = connector.run_experiments(qobj, **message.payload)
                channel.send(results_file)

            if message.type == KernelMessageType.REGISTER:
                connector.register_job(message.payload)


def connect() -> Tuple[mp.Process, Connection]:
    """Opens the connector and returns a Connection to interface with it with pickleble data

    Returns:
        the process and Connection to be used to communicate with the Connector
    """
    outer_end, inner_end = mp.Pipe()

    process = mp.Process(target=serve, args=(inner_end,))
    process.start()

    return process, outer_end


class Kernel:
    """The controller of the hardware"""

    setup = InstrumentCoordinator("tergite_kernel", add_default_generic_icc=False)

    # heap memory, so that instrument drivers do not get garbage collected
    shared_mem = list()

    def __init__(
        self,
        config_file: Union[str, bytes, os.PathLike],
    ):
        conf = KernelConfig.from_yaml(config_file)

        # Tell Quantify where to store data
        dh.set_datadir(conf.general.data_directory)

        self.quantify_config = conf.to_quantify()
        self.hardware_map = {
            clock: port
            for (port, clock), instrument in generate_port_clock_to_device_map(
                self.quantify_config
            ).items()
        }

        self.is_simulator = conf.general.is_simulator
        self.simulator = (
            None
            if not self.is_simulator
            else _SIMULATOR_MAP.get(conf.general.simulator_type, None)
        )

        # load clusters
        for cluster in conf.clusters:
            dummy_cfg: Optional[Dict[int, qblox_instruments.ClusterType]] = None
            if cluster.is_dummy:
                dummy_cfg = {
                    slot + 1: _QBLOX_CLUSTER_TYPE_MAP[module.instrument_type]
                    for slot, module in enumerate(cluster.modules)
                }
            # We only support qblox_instruments.Cluster for now. Pulsar and any other native interfaces were dropped
            # because they cause a chaotic configuration.
            # The Cluster was also the only one documented on quantify-scheduler docs at the time of the refactor
            # https://quantify-os.org/docs/quantify-scheduler/dev/reference/qblox/Cluster.html
            device = qblox_instruments.Cluster(
                name=cluster.name,
                identifier=None,
                dummy_cfg=dummy_cfg,
            )
            component = ClusterComponent(device)

            Kernel.shared_mem.append(device)
            rich.print(f"Instantiated Cluster driver for '{cluster.name}'")
            Kernel.setup.add_component(component)
            rich.print(
                f"Added '{component.name}' to instrument coordinator '{Kernel.setup.name}'"
            )

        # load generic QCoDes instruments
        for instrument in conf.generic_qcodes_instruments:
            # instantiate the device
            driver = import_python_object_from_string(
                instrument.instrument_driver.import_path
            )
            device = driver(**instrument.instrument_driver.kwargs)

            # set its parameters by calling them as commands
            # https://microsoft.github.io/Qcodes/examples/15_minutes_to_QCoDeS.html#Example-of-setting-and-getting-parameters
            for command, value in instrument.parameters.items():
                try:
                    qcodes_command = getattr(device, command)
                    qcodes_command(value)
                    rich.print(f"Set '{command}' to {value}")
                except (AttributeError, TypeError):
                    # ignore invalid parameters
                    pass

            Kernel.shared_mem.append(device)
            rich.print(
                f"Instantiated {instrument.instrument_driver.import_path.split('.')[-1]} driver for '{instrument.name}'"
            )

    def register_job(self, tag: str = ""):
        # TODO: The fields tuid, experiment_folder, and logger could be class properties
        self.tuid = gen_tuid()
        self.experiment_folder = Path(create_exp_folder(tuid=self.tuid, name=tag))
        self.logger = ExperimentLogger(self.tuid)

        self.logger.info(f"Registered job: {self.tuid}")
        self.logger.info(
            f"Loaded hardware configuration: {json.dumps(self.quantify_config, indent=4)}"
        )
        self.logger.info(
            f"Generated hardware map: {json.dumps(self.hardware_map, indent=4)}"
        )

    def run(self, experiment: Experiment, /):
        Kernel.setup.stop()

        # compile to hardware
        # TODO: Here, we can use the new @timer decorator in the benchmarking package
        t1 = datetime.now()
        if self.is_simulator:
            compiled_schedule = hardware_compile(
                schedule=experiment.schedule, hardware_cfg=self.quantify_config
            )
        else:
            absolute_timed_schedule = determine_absolute_timing(
                copy.deepcopy(experiment.schedule)
            )
            compiled_schedule = hardware_compile(
                schedule=absolute_timed_schedule, hardware_cfg=self.quantify_config
            )

        t2 = datetime.now()
        print(t2 - t1, "DURATION OF COMPILING")

        # log the sequencer assembler programs and the schedule timing table
        self.logger.log_Q1ASM_programs(compiled_schedule)
        self.logger.log_schedule(compiled_schedule)

        # upload schedule to instruments & arm sequencers
        Kernel.setup.prepare(compiled_schedule)

        # start experiment
        # TODO: Here, we can use the new @timer decorator from the benchmarking package
        t3 = datetime.now()
        Kernel.setup.start()

        # wait for program to finish and return acquisition
        # TODO: What is the return type of retrieve_acquisition()?
        results = Kernel.setup.retrieve_acquisition()
        print(f"{results=}")
        t4 = datetime.now()
        print(t4 - t3, "DURATION OF MEASURING")
        return results

    def simulate(self, experiment: Experiment, /):
        schedule = experiment.schedule
        compiled_sched = self.simulator.compile(schedule)

        self.logger.log_schedule(compiled_sched)

        return self.simulator.run(compiled_sched, output="voltage_single_shot")

    def construct_experiments(self, qobj: PulseQobj, /) -> list:
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

    def debug_save_qobj(self, qobj: PulseQobj):
        """Saves the incoming PulseQobj for debugging.
        This is a re-encoding when using the rest_api, but it is needed for local debugging.
        TODO: Avoid re-encoding for external jobs.
        """
        file = self.experiment_folder / "qobj.json"
        with open(file, mode="w") as qj:
            json.dump(qobj.to_dict(), qj, cls=PulseQobj_encoder, indent="\t")
        rich.print(f"Saved PulseQobj at {file}")
        self.logger.info(f"Saved PulseQobj at {file}")

    def run_experiments(
        self,
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
        self.debug_save_qobj(qobj)
        results_file_path: Optional[Path] = None
        try:
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
            results_file_path = self.experiment_folder / filename
            storage = StorageFile(
                results_file_path,
                mode="w",
                job_id=job_id,
                tuid=self.tuid,
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
                    desc=self.tuid,
                )
            ):
                print(datetime.now(), "IN RUN_EXPERIMENTS, START RUN")

                if self.is_simulator:
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
                ok_str := f"Completed {job_id if job_id else 'local job'} with tuid {self.tuid}."
            )
            self.logger.info(ok_str)

        # record exceptions
        except Exception as e:
            exc_str = f"\n{format_exc()}"
            if enable_traceback:
                rich.print(exc_str)
            self.logger.error(exc_str)

            rich.print(
                fail_str := f"Failed {job_id if job_id else 'local job'} with tuid {self.tuid}. Error: {repr(e)}"
            )
            self.logger.info(fail_str)
            raise e

        # cleanup, regardless if job failed or succeeded
        finally:
            try:
                storage.file.close()
            except UnboundLocalError:
                pass  # no storage to close

        return results_file_path


class KernelMessageType(int, enum.Enum):
    """The message types sent to this quantum kernel service"""

    CLOSE = 1
    RUN = 2
    REGISTER = 3


@dataclasses.dataclass
class KernelMessage:
    """Record schema of the messages sent to this quantum kernel service"""

    type: KernelMessageType
    payload: Any
