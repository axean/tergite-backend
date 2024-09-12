# This code is part of Tergite
#
# (C) Copyright Axel Andersson 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import functools
import pickle
from enum import Enum
from pathlib import Path
from typing import Literal, Union

import h5py
import numpy as np
import xarray as xr

from . import utils as parse


class MeasLvl(Enum):
    DISCRIMINATED = 2
    INTEGRATED = 1
    RAW = 0


class MeasRet(Enum):
    AVERAGED = 1
    APPENDED = 0


class RegisterOrder(Enum):
    LITTLE_ENDIAN = "little"
    BIG_ENDIAN = "big"


class RegisterSparsity(Enum):
    FULL = "full"
    SPARSE = "sparse"


class StorageFile:
    delimiter = "~"

    def __init__(
        self,
        # filepath in OS
        storage_file: Path,
        /,
        *,
        # mode specifier
        mode: Literal["r", "w"],
        # during write mode, the following need to be specified
        tuid: Union[str, None] = None,
        meas_return: Union[MeasRet, None] = None,
        meas_level: Union[MeasLvl, None] = None,
        meas_return_cols: Union[int, None] = None,
        # during write mode, the following are optional
        job_id: Union[str, None] = None,
        memory_slot_size: int = 100,  # the maximum number of classical register slots
    ):
        self.mode = "r" if str.lower(mode) in ("r", "read") else "w"
        self.file = h5py.File(storage_file, mode=self.mode)
        self.memory_slot_size = memory_slot_size

        if self.mode == "w":
            assert tuid is not None, "tuid needs to be specified during write mode"
            assert (
                meas_return is not None
            ), "meas_return needs to be specified during write mode"
            assert (
                meas_level is not None
            ), "meas_level needs to be specified during write mode"
            assert (
                meas_return_cols is not None
            ), "meas_return_cols needs to be specified during write mode"

            self.file.attrs["tuid"] = self.tuid = tuid
            self.file.attrs["meas_return"] = self.meas_return = meas_return.value
            self.file.attrs["meas_level"] = self.meas_level = meas_level.value
            self.file.attrs[
                "meas_return_cols"
            ] = self.meas_return_cols = meas_return_cols

            if job_id is not None:
                self.file.attrs["job_id"] = self.job_id = job_id
                self.local = False
            else:
                self.local = True

            self.header = self.file.create_group("header")
            self.experiments = self.file.create_group("experiments")

        else:
            self.tuid = self.file.attrs["tuid"]
            self.meas_return = MeasRet(self.file.attrs["meas_return"])
            self.meas_level = MeasLvl(self.file.attrs["meas_level"])
            self.meas_return_cols = self.file.attrs["meas_return_cols"]

            if "job_id" in self.file.attrs.keys():
                self.job_id = self.file.attrs["job_id"]
                self.local = False
            else:
                self.local = True

            self.header = self.file["header"]
            self.experiments = self.file["experiments"]

    # TODO: leave register_sparsity as full and set it to sparse for cases where there are multiple values per shot
    def as_readout(
        self: "StorageFile",
        discriminator: callable,
        *,
        disc_two_state: bool = False,
        register_order: str = "little",
        register_sparsity: str = "full",
    ) -> list:
        """
        Interpret measurement data from experiments as bitstrings.
        Requires a callable discriminator function.

        The discriminator has to be a Python function
        which takes two arguments "qubit_index" and "iq_point"
        and returns a binary value (0/1).

        Returns an object of type List[List[hex]]
        where the outer dimension is the experiment number.
        The inner dimension is the shot number.

        The hex value corresponds to how the classical slot register
        is read "from right to left" (Little endian) in binary. This
        binary value is then converted to hex (base 16).
        """

        # ------------ Helper functions
        def _register_full(register: list, *, slot_idxs: list) -> str:
            full_register = ["0" for _ in range(self.memory_slot_size)]
            for i, idx in enumerate(slot_idxs):
                full_register[self.memory_slot_size - 1 - idx] = str(register[i])
            return "".join(full_register)

        def _register_sparse(register: list, **kwargs) -> str:
            return "".join(register)

        def _map_to_hex(
            binary_transposed_msmt_matx, *, slot_idxs: list, register_parse_fn: callable
        ) -> list:
            le_registers = map(
                functools.partial(register_parse_fn, slot_idxs=slot_idxs),
                binary_transposed_msmt_matx.astype(str),
            )
            b_int = 3 if disc_two_state else 2
            hex_le_registers = map(lambda reg: hex(int(reg, b_int)), le_registers)
            return list(hex_le_registers)

        # ------------ Normalize arguments
        register_sparsity = str.strip(str.lower(register_sparsity))
        register_order = str.strip(str.lower(register_order))

        # ------------ Select if register readout should be sparse or not
        if RegisterSparsity(register_sparsity) == RegisterSparsity.SPARSE:
            register_parse_fn = _register_sparse
        elif RegisterSparsity(register_sparsity) == RegisterSparsity.FULL:
            register_parse_fn = _register_full
        else:
            raise ValueError(f"Invalid register sparsity setting: {register_sparsity}.")

        # ------------ Select register bit reading order
        if RegisterOrder(register_order) == RegisterOrder.LITTLE_ENDIAN:
            register_reverse = True
        elif RegisterOrder(register_order) == RegisterOrder.BIG_ENDIAN:
            register_reverse = False
        else:
            raise ValueError(f"Invalid register order setting: {register_order}.")

        # ------------ Discriminate the I/Q readout
        assert callable(discriminator)
        readout = list()
        for tag, experiment_data in self.sort_items(self.experiments.items()):
            memory = list()

            slots = filter(lambda item: "slot" in item[0], experiment_data.items())

            # keep track of which classical register slots were used
            # these will be in little endian order
            slot_idxs = list()

            # sort in reverse for little endian (descending slot order)
            for slot_tag, slot_data in self.sort_items(slots, reverse=register_reverse):
                # TODO: assert if measurement shape if equal to the number of shots
                # assert (
                #     slot_data["measurement"].shape[0] == 1
                # ), "Max one acquisition per channel for word readout."
                kets = np.zeros(slot_data["measurement"].shape[0]).astype(int)
                slot_idx = int(slot_tag.split(self.delimiter)[1])
                slot_idxs.append(slot_idx)
                row = slot_data["measurement"][0, :]
                kets = discriminator(qubit_idx=slot_idx, iq_points=row)
                memory.append(kets)
            # binary matrix where rows are classical register values
            # and columns are register value per shot
            memory = np.asarray(memory)
            # transposing the memory we get a binary matrix where columns are
            # classical register values and rows are value per shot
            memory = np.transpose(memory)
            # get hexlist of classical registers for every shot
            # and append to readout list
            readout.append(
                _map_to_hex(
                    memory, slot_idxs=slot_idxs, register_parse_fn=register_parse_fn
                )
            )

        return readout

    def as_xarray(self: "StorageFile") -> xr.Dataset:
        """Attempts to parse the storage file as an N-dimensional parametric sweep.
        Returns an xarray dataset. Automatically detects multiplexed sweeps.

        Assumes that the sweep order of the independent variables is in the order
        of main_dims, e.g.
            if main_dims = ("frequencies", "amplitudes"),
            then assumes index order is:
                for f in frequencies:
                    for a in amplitudes:
                        ...
        """
        if (self.meas_return == MeasRet.APPENDED) and (
            self.meas_level == MeasLvl.DISCRIMINATED
        ):
            return NotImplemented  # discriminator?

        elif (self.meas_return == MeasRet.AVERAGED) and (
            self.meas_level == MeasLvl.DISCRIMINATED
        ):
            return NotImplemented  # discriminator?

        elif (self.meas_return == MeasRet.APPENDED) and (
            self.meas_level == MeasLvl.INTEGRATED
        ):
            return parse.appended_integrated(data=self)

        elif (self.meas_return == MeasRet.AVERAGED) and (
            self.meas_level == MeasLvl.INTEGRATED
        ):
            return parse.averaged_integrated(data=self)

        elif (self.meas_return == MeasRet.AVERAGED) and (
            self.meas_level == MeasLvl.RAW
        ):
            return parse.averaged_raw(data=self)

        else:
            raise NotImplementedError(
                f"Invalid storage file metadata: {self.meas_return} and {self.meas_level} is not implemented."
            )

    # ------------------------------------------------------------------------
    def get_experiment(self: "StorageFile", name: str):
        """Returns an experiment group in the file, if it exists.
        If it does not exist, makes a new experiment group and returns that.
        """
        if name not in self.experiments.keys():
            experiment = self.experiments.create_group(name)
        else:
            experiment = self.experiments[name]

        return experiment

    @functools.cached_property
    def sorted_measurements(self: "StorageFile") -> list:
        # TODO: this would only find files for simulated readout output
        return sorted(
            parse.find(self.experiments, "measurement"),
            key=lambda path: path[0].split(self.delimiter)[1],
        )

    # ------------------------------------------------------------------------
    def store_metadata(
        self: "StorageFile",
        grpkey: str,
        key: str,
        tmp_data: dict,
        DEFAULT_VALUE: object = "",
    ):
        """Helper function which stores metadata in a specific HDF group or dataset in this file."""
        # retrieve referred group or dataset in HDF file
        grp = self.file[grpkey]
        # if the specified metadata exists
        if key in tmp_data.keys():
            # store metadata as an attribute for that group or dataset
            grp.attrs[key] = tmp_data[key]
        else:
            print(f"Failed to store {key}")
            grp.attrs[key] = DEFAULT_VALUE

    def store_qobj_header(self: "StorageFile", qobj_header: dict):
        """Stores metadata about the experiment from the qobj header."""
        header_grp = self.header.create_group("qobj")

        if "backend_name" in qobj_header:
            backend_grp = header_grp.create_group("backend")
            self.store_metadata("header/qobj/backend", "backend_name", qobj_header)

        if "sweep" in qobj_header:
            sweep_grp = header_grp.create_group("sweep")

            self.store_metadata(
                "header/qobj/sweep", "dataset_name", qobj_header["sweep"]
            )
            self.store_metadata(
                "header/qobj/sweep", "serial_order", qobj_header["sweep"]
            )
            self.store_metadata(
                "header/qobj/sweep", "batch_size", qobj_header["sweep"], DEFAULT_VALUE=1
            )

            for path in parse.find(qobj_header["sweep"], "slots"):
                # create group for storage
                slots_grp = sweep_grp.create_group("/".join(path))

                param = path[
                    -2
                ]  # -1 is "slots", -2 is parameter name, -3 is "parameters"
                self.store_metadata(
                    f"header/qobj/sweep/parameters/{param}",
                    "long_name",
                    qobj_header["sweep"]["parameters"][param],
                )
                self.store_metadata(
                    f"header/qobj/sweep/parameters/{param}",
                    "unit",
                    qobj_header["sweep"]["parameters"][param],
                )

                # traverse qobj_header["sweep"] until slots dict
                slots_data = qobj_header["sweep"]
                for k in path:
                    slots_data = slots_data[k]

                # store all specified sweep parameter data in respective HDF datasets
                for slot_idx, slot_data in slots_data.items():
                    data = np.asarray(slot_data)
                    slots_grp.create_dataset(
                        f"slot{StorageFile.delimiter}{slot_idx}",
                        shape=data.shape,
                        dtype=data.dtype,
                    )
                    slots_grp[f"slot{StorageFile.delimiter}{slot_idx}"][...] = data

    def store_graph(self: "StorageFile", graph: object, name: str):
        """Store the bytes of an experiment's graph into its experiment group.
        This graph can be loaded with StorageFile.read_graph.
        # TODO: I cannot find any place where this read_graph is implemented and I do not see why we would need it
        """
        experiment = self.get_experiment(name)
        blob = pickle.dumps(graph)
        g = experiment.create_dataset(
            "experiment_graph", shape=len(blob), dtype=np.ubyte
        )
        g[...] = np.frombuffer(blob, dtype=np.ubyte)

    def store_experiment_data(self: "StorageFile", *, experiment_data: dict, name: str):
        """Store the data of an acquisition into an experiment group."""
        experiment = self.get_experiment(name)

        for acq_index, acq in enumerate(experiment_data["data_vars"]):
            ch = f"slot{StorageFile.delimiter}{acq}"
            # Get maximum acquisition index in each acquisition channel
            max_acq_idx = experiment_data["dims"][f"acq_index_{acq_index}"]

            # For each acqusition channel, create a corresponding measurement matrix
            # in a memory slot whose index corresponds to the acqusition channel,
            # unless it already exists.
            if ch not in experiment.keys():
                channel = experiment.create_group(ch)

                # The columns of the matrix are the # of shots and the rows are
                # the horizontally composed measurements on the acqusition channel
                # (row indices corresponding to acquisition indices)
                print(f"creating data set for {ch}")
                channel.create_dataset(
                    "measurement",
                    shape=(max_acq_idx, self.meas_return_cols),
                    dtype=complex,
                )

        for acq, data in experiment_data["data_vars"].items():
            ch = f"slot{StorageFile.delimiter}{acq}"

            tmp = np.zeros(len(data["data"]), dtype=complex)

            for idx in range(len(data["data"])):
                tmp.real[idx] = np.real(data["data"][idx])
                tmp.imag[idx] = np.imag(data["data"][idx])

            experiment[ch]["measurement"][...] = tmp

    # ------------------------------------------------------------------------

    @classmethod
    def sort_items(cls: "StorageFile", items: list, reverse: bool = False) -> iter:
        return sorted(
            items, key=lambda tup: int(tup[0].split(cls.delimiter)[1]), reverse=reverse
        )

    @staticmethod
    def sanitized_name(users_experiment_name: str, experiment_index: int):
        """Return a cleaned version of a given experiment name."""
        name = "".join(
            x for x in users_experiment_name if x.isalnum() or x in " -_,.()"
        )
        return f"{name}{StorageFile.delimiter}{experiment_index}"
