# This code is part of Tergite
#
# (C) Stefan Hill (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import abc
import json
from datetime import datetime
from pathlib import Path
from traceback import format_exc
from typing import Optional

import numpy as np
import rich
import xarray
from qiskit.providers.ibmq.utils.json_encoder import IQXJsonEncoder as PulseQobj_encoder
from qiskit.qobj import PulseQobj
from quantify_core.data import handling as dh
from quantify_core.data.handling import create_exp_folder, gen_tuid
from tqdm import tqdm

import settings
from app.libs.quantum_executor.base.experiment import BaseExperiment
from app.libs.quantum_executor.utils.instruction import meas_settings
from app.libs.quantum_executor.utils.logger import ExperimentLogger
from app.libs.storage_file import StorageFile


class QuantumExecutor(abc.ABC):
    def __init__(self):
        dh.set_datadir(settings.EXECUTOR_DATA_DIR)

    def register_job(self, tag: str = ""):
        # TODO: The fields tuid, experiment_folder, and logger could be class properties
        self.tuid = gen_tuid()
        self.experiment_folder = Path(create_exp_folder(tuid=self.tuid, name=tag))
        self.logger = ExperimentLogger(self.tuid)
        self.logger.info(f"Registered job: {self.tuid}")

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

    @abc.abstractmethod
    def construct_experiments(self, qobj: PulseQobj, /):
        pass

    @abc.abstractmethod
    def run(self, experiment: BaseExperiment, /) -> xarray.Dataset:
        pass

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

                experiment_data = self.run(experiment)

            
                # avoid experiment structure for simulation 
                # TODO: consider overriding this method in qiskit_executor
                # TODO: make more appropriate naming value and key
                import uuid 

                if self.backend.backend_name == "fake_openpulse_1q":
                    storage.store_experiment_array(
                        experiment_data=experiment_data,
                        name="temp_dummy_name-%s" % uuid.uuid4()
                    )
                else:
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

    @abc.abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
