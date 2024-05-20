# This code is part of Tergite
#
# (C) Nicklas Botö, Fabian Forslund 2023
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

"""The Simulator based on the proprietary SCQT (superconducting qubit tools) tools"""
from datetime import datetime
from functools import reduce
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from xarray import DataArray, Dataset

try:
    from superconducting_qubit_tools.device_under_test.mock_edge import (
        MockSuddenNetZeroEdge,
    )
    from superconducting_qubit_tools.device_under_test.mock_setup import (
        create_five_transmon_mock_setup,
        set_calibrated_device_params,
        set_calibration_initial_settings,
    )
    from superconducting_qubit_tools.device_under_test.quantum_device_mock import (
        QuantumDeviceMock,
    )
    from superconducting_qubit_tools.device_under_test.transmock import (
        MockQubit,
        MockResonator,
        Transmock,
    )
    from superconducting_qubit_tools.initialization_scripts.hardware_configurations import (
        mock_hardware_cfg,
    )
    from superconducting_qubit_tools.instrument_coordinator.ins_coord_transmock import (
        InstrumentCoordinatorTransmock,
    )
    from superconducting_qubit_tools.instruments.dummy_instruments import (
        DummyFluxCurrent,
    )
    from superconducting_qubit_tools.simulation.executors import exec_schedule
except ImportError:
    # set dummies on all scqt classes
    exec_schedule = MagicMock()
    QuantumDeviceMock = MagicMock()
    MockSuddenNetZeroEdge = MagicMock()
    InstrumentCoordinatorTransmock = MagicMock()
    DummyFluxCurrent = MagicMock()
    Transmock = MagicMock()
    MockQubit = MagicMock()
    MockResonator = MagicMock()
    create_five_transmon_mock_setup = MagicMock()
    set_calibration_initial_settings = MagicMock()
    set_calibrated_device_params = MagicMock()
    mock_hardware_cfg = MagicMock()


from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import IdlePulse
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.schedule import CompiledSchedule, Schedule

from .base import BaseSimulator

N_QUBITS = 5
PORT_MAP = {f"drive{n}": f"q{n}:mw" for n in range(N_QUBITS)} | {
    f"readout{n}": f"q{n}:res" for n in range(N_QUBITS)
}
CLOCK_MAP = {f"d{n}": f"q{n}.01" for n in range(N_QUBITS)} | {
    f"m{n}": f"q{n}.ro" for n in range(N_QUBITS)
}
CLOCK_PORT_MAP = {f"d{n}": f"drive{n}" for n in range(N_QUBITS)} | {
    f"m{n}": f"readout{n}" for n in range(N_QUBITS)
}
QUBIT_IX_MAP = {f"q{n}": n for n in range(N_QUBITS)}
PORT_QUBIT_MAP = {f"q{n}:res": f"q{n}" for n in range(N_QUBITS)}


class Simulator(BaseSimulator):
    def __init__(self: object, /) -> None:
        qubit_names = [f"q{n}" for n in range(N_QUBITS)]
        edge_names = ["edge_q2_q0", "edge_q2_q1", "edge_q3_q2", "edge_q4_q2"]

        setup = create_five_transmon_mock_setup()

        self.device: QuantumDeviceMock = setup.pop("quantum_device")
        self.instrument_coordinator: InstrumentCoordinatorTransmock = setup.pop(
            "instrument_coordinator"
        )
        self._flux_current: DummyFluxCurrent = setup.pop("flux_current")
        self.qubits: list[Transmock] = [setup.pop(q) for q in qubit_names]
        self.edges: list[MockSuddenNetZeroEdge] = [setup.pop(e) for e in edge_names]
        self.compiler: SerialCompiler = SerialCompiler("sim-compiler", self.device)

        self.instrument_coordinator.mode("simulate")
        self.device.hardware_config(mock_hardware_cfg)
        set_calibration_initial_settings(qubit_names)
        set_calibrated_device_params(qubit_names)

        # Fix for measurement issues in SCQT product
        for i, qubit in enumerate(self.qubits):
            qubit.measure.acq_channel(i)
            qubit.hardware_options.attenuations.mw_output_att(0)
            qubit.hardware_options.attenuations.ro_output_att(36)

    def run(
        self: object,
        schedule: CompiledSchedule,
        /,
        *,
        output: str = "voltage_single_shot",
    ) -> Dataset:
        """Runs a given compiled schedule on the simulator.

        Args:
            schedule (Schedule): The schedule to run.
            output (str, optional): Mode of output, either 'voltage' which returns complex voltages,
                or 'raw' which returns measurement probabilites. Defaults to 'voltage_single_shot'.
        """
        sim_data = exec_schedule(schedule=schedule, device=self.device, output=output)
        return sim_data

    def compile(self: object, schedule: Schedule, /) -> CompiledSchedule:
        """Compiles a schedule which has been translated by TQC to be run on the simulator.

        Args:
            schedule (Schedule): The schedule to compile.
        """
        schedulables = schedule.schedulables
        operations = schedule.operations
        schedulables_sorted = sorted(
            schedulables.items(), key=lambda op: op[1].data["abs_time"]
        )

        compiled_schedule = Schedule(schedule.name, schedule.repetitions)

        for schedulable_id, schedulable in schedulables_sorted:
            op_id = schedulable.data["operation_repr"]
            op = operations[op_id]
            op_name, *_ = op.name.split("-")

            timing_constraints, *_ = schedulable.data["timing_constraints"]
            ref_id = timing_constraints["ref_schedulable"]
            ref_pt_new = timing_constraints["ref_pt_new"]
            ref_pt = timing_constraints["ref_pt"]
            rel_time = timing_constraints["rel_time"]
            ref_schedulable = compiled_schedule.schedulables.get(ref_id, None)

            if op_name in {"setf", "setp", "shiftf", "fc", "initial_object"}:
                compiled_schedule.add(
                    IdlePulse(duration=0),
                    rel_time=rel_time,
                    ref_op=ref_schedulable,
                    ref_pt=ref_pt,
                    ref_pt_new=ref_pt_new,
                    label=schedulable_id,
                )
                continue

            updated_op = _update_op_port_clock(op)

            if op_name in {"constant", "delay"}:
                pulse_info, *_ = updated_op["pulse_info"]
                duration = pulse_info["duration"]
                port = pulse_info["port"]
                clock = pulse_info["clock"]

                compiled_schedule.add(
                    _delay(duration, port, clock),
                    rel_time=rel_time,
                    ref_op=ref_schedulable,
                    ref_pt=ref_pt,
                    ref_pt_new=ref_pt_new,
                    label=schedulable_id,
                )

            elif op_name == "SSBIntegrationComplex":
                acq_info, *_ = op["acquisition_info"]
                port = acq_info["port"]
                updated_op.update({"gate_info": {"qubits": [PORT_QUBIT_MAP[port]]}})
                compiled_schedule.add(
                    updated_op,
                    rel_time=rel_time,
                    ref_op=ref_schedulable,
                    ref_pt=ref_pt,
                    ref_pt_new=ref_pt_new,
                    label=schedulable_id,
                )

            else:
                compiled_schedule.add(
                    updated_op,
                    rel_time=rel_time,
                    ref_op=ref_schedulable,
                    ref_pt=ref_pt,
                    ref_pt_new=ref_pt_new,
                    label=schedulable_id,
                )

        for _, resource in schedule.resources.items():
            if resource.name.split("-")[0] not in {"cl0.baseband"}:
                compiled_schedule.add_resource(_update_resource_name(resource))

        # compilation.determine_absolute_timing(compiled_schedule)
        return self.compiler.compile(compiled_schedule)

    def generate_backend_config(self: object, /) -> dict[str, Any]:
        """
        Returns the device backend config for the backend class in tergite.qiskit client
        """
        qubits: list[str] = self.device.elements()

        backend_dict = {}
        device_properties = {}

        backend_dict["name"] = "SimulatorA"
        backend_dict["characterized"] = True
        backend_dict["open_pulse"] = True
        backend_dict["timelog"] = {"REGISTERED": datetime.today().isoformat()}
        backend_dict["version"] = datetime.today().strftime("%Y.%m.%d")
        backend_dict["num_qubits"] = len(qubits)
        backend_dict["num_couplers"] = 2 * len(self.edges)
        backend_dict["num_resonators"] = len(qubits)
        backend_dict["dt"] = 1e-09
        backend_dict["dtm"] = 1e-09
        backend_dict["meas_map"] = [list(range(len(qubits)))]
        backend_dict["coupling_map"] = reduce(
            lambda es, e: es
            + [
                [self.qubits.index(e.qubit_child), self.qubits.index(e.qubit_parent)],
                [self.qubits.index(e.qubit_parent), self.qubits.index(e.qubit_child)],
            ],
            map(lambda edge: edge.mock_edge, self.edges),
            [],
        )

        device_properties["qubit"] = []
        device_properties["readout_resonator"] = []

        for i, transmon in enumerate(self.qubits):
            qubit: MockQubit = transmon.mock_qubit
            resonator: MockResonator = transmon.mock_resonator

            qubit_dict = {}
            resonator_dict = {}

            qubit_dict["index"] = i
            qubit_dict["frequency"] = transmon.clock_freqs.f01()

            # calibrated values
            if i == 0:
                qubit_dict["pi_pulse_amplitude"] = 0.17555712637424228
            if i == 1:
                qubit_dict["pi_pulse_amplitude"] = 0.17535338530538067
            if i == 2:
                qubit_dict["pi_pulse_amplitude"] = 0.17873594718151276
            if i == 3:
                qubit_dict["pi_pulse_amplitude"] = 0.17326197853513559
            if i == 4:
                qubit_dict["pi_pulse_amplitude"] = 0.16948867103728774

            qubit_dict["pi_pulse_duration"] = 5.6e-08  # calibrated
            qubit_dict["pulse_sigma"] = 5.6e-08 / 8

            qubit_dict["pulse_type"] = "Gaussian"
            qubit_dict["t1_decoherence"] = qubit.t1()
            qubit_dict["t2_decoherence"] = qubit.t2_star()

            device_properties["qubit"].append(qubit_dict)

            resonator_dict["index"] = i
            resonator_dict["acq_delay"] = transmon.measure.acq_delay()
            resonator_dict["acq_integration_time"] = transmon.measure.integration_time()
            resonator_dict["frequency"] = resonator.resonant_frequency_bare()
            resonator_dict[
                "pulse_amplitued"
            ] = transmon.measure.pulse_amp()  # faulty spelling to match qiskit
            resonator_dict["pulse_delay"] = transmon.measure.pulse_delay()
            resonator_dict["pulse_duration"] = transmon.measure.pulse_duration()
            resonator_dict["pulse_type"] = "Square"

            device_properties["readout_resonator"].append(resonator_dict)

        device_properties["coupler"] = [{}]

        backend_dict["device_properties"] = device_properties
        backend_dict["meas_lo_freq"] = [None for _ in range(len(qubits))]
        backend_dict["qubit_lo_freq"] = [None for _ in range(len(qubits))]

        return backend_dict


def _delay(duration: float, port: str, clock: str, /) -> Operation:
    delay_dict = {
        "name": "Idle",
        "pulse_info": [
            {
                "wf_func": None,
                "t0": 0,
                "duration": duration,
                "clock": clock,
                "port": port,
            }
        ],
    }
    delay = IdlePulse(duration=duration)
    delay.data.update(delay_dict)
    delay._update()
    return delay


def _update_op_port_clock(op: Operation, /) -> Operation:
    """
    Updates operation port and clock names from the convention used in
    the backend class of tergite-qiskit-connector to that used by SCQT
    """
    for pulse in op["pulse_info"]:
        clock = pulse["clock"]
        port = pulse.get("port") or CLOCK_PORT_MAP[pulse.get("clock")]
        pulse.update({"clock": CLOCK_MAP[clock]})
        pulse.update({"port": PORT_MAP[port]})

    for acq_data in op["acquisition_info"]:
        port = acq_data["port"]
        clock = acq_data["clock"]
        acq_data.update({"port": PORT_MAP[port], "clock": CLOCK_MAP[clock]})
        for waveform in acq_data["waveforms"]:
            port = waveform["port"]
            clock = waveform["clock"]
            waveform.update({"port": PORT_MAP[port], "clock": CLOCK_MAP[clock]})

    return op


def _update_resource_name(channel: ClockResource, /) -> ClockResource:
    """
    Updates channel name from the convention used in the backend class
    of tergite-qiskit-connector to that used by SCQT
    """
    channel.update({"name": CLOCK_MAP[channel.name]})

    return channel


def _fetch_acquisition_metadata(
    schedule: CompiledSchedule,
) -> list[tuple[int, int, int]]:
    acquisitions = [
        op["acquisition_info"][0]
        for _, op in schedule.operations.items()
        if op["acquisition_info"]
    ]

    return [
        (i, acq_info["acq_channel"], acq_info["acq_index"])
        for i, acq_info in enumerate(acquisitions)
    ]


def _format_output(sim_data: DataArray, output: str) -> Dataset:
    """
    Formats the simulation data in an output coherent with the live hardware.

    Args:
        sim_data (DataArray): Simulator output data
        output (str): Chosen simulator output format (only voltage_single_shot currently tested)
    Returns:
        dataset (Dataset)
    """
    schedule_qubits = [QUBIT_IX_MAP[qbit] for qbit in sim_data.qubit.data]

    return Dataset(
        dict(
            [
                qbit_ix,
                (
                    [f"acq_index_{qbit_ix}"]
                    if output == "voltage"
                    else ["repetitions", f"acq_index_{qbit_ix}"],
                    np.vstack((*[sim_data[i, :, i].data],)).T,
                ),
            ]
            for i, qbit_ix in enumerate(schedule_qubits)
        )
    )
