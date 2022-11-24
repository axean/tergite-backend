# This code is part of Tergite
#
# (C) Copyright Simon Genne, Arvid Holmqvist, Bashar Oumari, Jakob Ristner,
#               Björn Rosengren, and Jakob Wik 2022 (BSc project)
# (C) Copyright Fabian Forslund, Nicklas Botö 2022
# (C) Copyright Abdullah-Al Amin 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from backend_properties_updater.backend_properties_updater_utils import (
    TYPE1,
    TYPE2,
    TYPE4_DOMAIN,
    init_NDUV,
    update_NDUV,
)

UPPER_FREQUENCY_LIMIT = 5e9
LOWER_FREQUENCY_LIMIT = 4e9
FREQUENCY_TYPES = [TYPE1, TYPE4_DOMAIN]

UPPER_ANHARMONICITY_LIMIT = -200e6
LOWER_ANHARMONICITY_LIMIT = -300e6
ANHARMONICITY_TYPES = [TYPE1]

UPPER_CHI_SHIFT_LIMIT = 5e6
LOWER_CHI_SHIFT_LIMIT = 1e6
CHI_SHIFT_TYPES = [TYPE1]

UPPER_G_RQ_LIMIT = 100e6
LOWER_G_RQ_LIMIT = 50e6
G_RQ_TYPES = [TYPE1]

UPPER_QUBIT_TPURCELL_LIMIT = 1e10
LOWER_QUBIT_TPURCELL_LIMIT = 1e-3
QUBIT_TPURCELL_TYPES = [TYPE1]

UPPER_QUBIT_TDRIVE_LIMIT = 1e10
LOWER_QUBIT_TDRIVE_LIMIT = 1e-3
QUBIT_TDRIVE_TYPES = [TYPE1]

UPPER_J_QC_LIMIT = 100e6
LOWER_J_QC_LIMIT = 50e6
J_QC_TYPES = [TYPE1]

UPPER_QUBIT_T1_LIMIT = 1e-4
LOWER_QUBIT_T1_LIMIT = 3e-5
QUBIT_T1_UPDATE_RATE = 20
QUBIT_T1_MAX_DELTA = 1e-6
QUBIT_T1_TYPES = [TYPE1, TYPE2, TYPE4_DOMAIN]

UPPER_QUBIT_T2_STAR_LIMIT = 2e-4
LOWER_QUBIT_T2_STAR_LIMIT = 3e-5
QUBIT_T2_STAR_UPDATE_RATE = 21
QUBIT_T2_STAR_MAX_DELTA = 2e-6
QUBIT_T2_STAR_TYPES = [TYPE1, TYPE2, TYPE4_DOMAIN]

UPPER_QUBIT_T_PHI_LIMIT = 200e-6
LOWER_QUBIT_T_PHI_LIMIT = 30e-6
QUBIT_T_PHI_UPDATE_RATE = 22
QUBIT_T_PHI_MAX_DELTA = 1e-5
QUBIT_T_PHI_TYPES = [TYPE1, TYPE2, TYPE4_DOMAIN]

UPPER_ASSIGNMENT_ERROR_GE_LIMIT = 0.1
LOWER_ASSIGNMENT_ERROR_GE_LIMIT = 0.01
ASSIGNMENT_ERROR_GE_UPDATE_RATE = 23
ASSIGNMENT_ERROR_GE_MAX_DELTA = 0.001
ASSIGNMENT_ERROR_GE_TYPES = [TYPE1]

UPPER_ASSIGNMENT_ERROR_GEF_LIMIT = 0.15
LOWER_ASSIGNMENT_ERROR_GEF_LIMIT = 0.01
ASSIGNMENT_ERROR_GEF_UPDATE_RATE = 24
ASSIGNMENT_ERROR_GEF_MAX_DELTA = 0.001
ASSIGNMENT_ERROR_GEF_TYPES = [TYPE1]

UPPER_READ_ERR_LIMIT = 0.1
LOWER_READ_ERR_LIMIT = 0.01
READ_ERR_UPDATE_RATE = 25
READ_ERR_MAX_DELTA = 0.001
READ_ERR_TYPES = [TYPE1]
READ_ERR_PROPS = [
    "read_err_prep0_meas1_ge",
    "read_err_prep1_meas0_ge",
    "read_err_prep0_meas1_gef",
    "read_err_prep0_meas2_gef",
    "read_err_prep0_meas2_gef",
    "read_err_prep1_meas0_gef",
    "read_err_prep1_meas2_gef",
    "read_err_prep2_meas0_gef",
    "read_err_prep2_meas1_gef",
]


class Qubit:
    def __init__(self, qubit_config: dict, coupling_map):
        self.qubit_config = qubit_config
        self.read_errs = _init_read_errs()
        self.J_qc = _init_J_qc(qubit_config["id"], coupling_map)
        self.frequency = init_NDUV(
            LOWER_FREQUENCY_LIMIT,
            UPPER_FREQUENCY_LIMIT,
            "frequency",
            "Hz",
            FREQUENCY_TYPES,
        )

        self.chi_shift = init_NDUV(
            LOWER_CHI_SHIFT_LIMIT,
            UPPER_CHI_SHIFT_LIMIT,
            "chi_shift",
            "Hz",
            CHI_SHIFT_TYPES,
        )
        self.g_rq = init_NDUV(
            LOWER_G_RQ_LIMIT,
            UPPER_G_RQ_LIMIT,
            "g_rq",
            "Hz",
            G_RQ_TYPES,
        )
        self.qubit_Tpurcell = init_NDUV(
            LOWER_QUBIT_TPURCELL_LIMIT,
            UPPER_QUBIT_TPURCELL_LIMIT,
            "qubit_Tpurcell",
            "s",
            QUBIT_TPURCELL_TYPES,
        )
        self.qubit_Tdrive = init_NDUV(
            LOWER_QUBIT_TDRIVE_LIMIT,
            UPPER_QUBIT_TDRIVE_LIMIT,
            "qubit_Tdrive",
            "s",
            QUBIT_TDRIVE_TYPES,
        )
        self.qubit_t1 = init_NDUV(
            LOWER_QUBIT_T1_LIMIT,
            UPPER_QUBIT_T1_LIMIT,
            "qubit_T1",
            "Hz",
            QUBIT_T1_TYPES,
        )
        self.qubit_t2_star = init_NDUV(
            LOWER_QUBIT_T2_STAR_LIMIT,
            UPPER_QUBIT_T2_STAR_LIMIT,
            "qubit_T2_star",
            "Hz",
            QUBIT_T2_STAR_TYPES,
        )
        self.qubit_T_phi = init_NDUV(
            LOWER_QUBIT_T_PHI_LIMIT,
            UPPER_QUBIT_T_PHI_LIMIT,
            "qubit_T_phi",
            "",
            QUBIT_T_PHI_TYPES,
        )
        self.assignment_error_ge = init_NDUV(
            LOWER_ASSIGNMENT_ERROR_GE_LIMIT,
            UPPER_ASSIGNMENT_ERROR_GE_LIMIT,
            "assignment_error_ge",
            "",
            ASSIGNMENT_ERROR_GE_TYPES,
        )
        self.assignment_error_gef = init_NDUV(
            LOWER_ASSIGNMENT_ERROR_GEF_LIMIT,
            UPPER_ASSIGNMENT_ERROR_GEF_LIMIT,
            "assignment_error_gef",
            "",
            ASSIGNMENT_ERROR_GEF_TYPES,
        )

    def update(self) -> None:
        self.read_errs = _update_read_errs(self.read_errs)
        self.qubit_t1 = update_NDUV(
            self.qubit_t1,
            QUBIT_T1_UPDATE_RATE,
            LOWER_QUBIT_T1_LIMIT,
            UPPER_QUBIT_T1_LIMIT,
            QUBIT_T1_MAX_DELTA,
        )
        self.qubit_t2_star = update_NDUV(
            self.qubit_t2_star,
            QUBIT_T2_STAR_UPDATE_RATE,
            LOWER_QUBIT_T2_STAR_LIMIT,
            UPPER_QUBIT_T2_STAR_LIMIT,
            QUBIT_T2_STAR_MAX_DELTA,
        )
        self.qubit_T_phi = update_NDUV(
            self.qubit_T_phi,
            QUBIT_T_PHI_UPDATE_RATE,
            LOWER_QUBIT_T_PHI_LIMIT,
            UPPER_QUBIT_T_PHI_LIMIT,
            QUBIT_T_PHI_MAX_DELTA,
        )
        self.assignment_error_ge = update_NDUV(
            self.assignment_error_ge,
            ASSIGNMENT_ERROR_GE_UPDATE_RATE,
            LOWER_ASSIGNMENT_ERROR_GE_LIMIT,
            UPPER_ASSIGNMENT_ERROR_GE_LIMIT,
            ASSIGNMENT_ERROR_GE_MAX_DELTA,
        )
        self.assignment_error_gef = update_NDUV(
            self.assignment_error_gef,
            ASSIGNMENT_ERROR_GEF_UPDATE_RATE,
            LOWER_ASSIGNMENT_ERROR_GEF_LIMIT,
            UPPER_ASSIGNMENT_ERROR_GEF_LIMIT,
            ASSIGNMENT_ERROR_GEF_MAX_DELTA,
        )

    def to_dict(self) -> dict:
        static_properties = [
            self.frequency.to_dict(),
            self.chi_shift.to_dict(),
            self.g_rq.to_dict(),
            self.qubit_Tpurcell.to_dict(),
            self.qubit_Tdrive.to_dict(),
            *self.J_qc,
        ]

        read_errs = [read_err.to_dict() for read_err in self.read_errs]
        dynamic_properties = [
            self.qubit_t1.to_dict(),
            self.qubit_t2_star.to_dict(),
            self.qubit_T_phi.to_dict(),
            self.assignment_error_ge.to_dict(),
            *read_errs,
        ]
        return {
            **self.qubit_config,
            **{
                "dynamic_properties": dynamic_properties,
                "static_properties": static_properties,
            },
        }


def _init_read_errs():
    read_errs = map(
        lambda prop: init_NDUV(
            LOWER_READ_ERR_LIMIT,
            UPPER_READ_ERR_LIMIT,
            prop,
            "",
            READ_ERR_TYPES,
        ),
        READ_ERR_PROPS,
    )
    return list(read_errs)


def _update_read_errs(read_errs):
    updated_read_errs = map(
        lambda nduv: update_NDUV(
            nduv,
            READ_ERR_UPDATE_RATE,
            LOWER_READ_ERR_LIMIT,
            UPPER_READ_ERR_LIMIT,
            READ_ERR_MAX_DELTA,
        ),
        read_errs,
    )
    return list(updated_read_errs)


def _init_J_qc(qubit_id: int, coupling_map):
    J_qcs = []
    neighbouring_qubits = _get_neighbouring_qubits(qubit_id, coupling_map)

    for coupler in coupling_map:
        coupled_qubits = coupler["qubits"]
        property_name = "J_qc_{" + str(qubit_id) + "," + str(coupler["id"]) + "}"

        if qubit_id in coupled_qubits:
            nduv = init_NDUV(
                LOWER_J_QC_LIMIT, UPPER_J_QC_LIMIT, property_name, "Hz", J_QC_TYPES
            )
        elif any(  # If any of the connected qubits are neighbours of this qubit.
            coupled_qubit in neighbouring_qubits for coupled_qubit in coupled_qubits
        ):
            nduv = init_NDUV(
                0, UPPER_J_QC_LIMIT * 1e-8, property_name, "Hz", J_QC_TYPES
            )
        else:
            nduv = init_NDUV(
                0, UPPER_J_QC_LIMIT * 1e-10, property_name, "Hz", J_QC_TYPES
            )
        J_qcs.append(nduv.to_dict())

    return J_qcs


def _get_neighbouring_qubits(qubit_id, coupling_map):
    neighbours = []

    for coupler in coupling_map:
        for connected_qubit in coupler["qubits"]:
            if connected_qubit == qubit_id:
                new_neighbours = [q for q in coupler["qubits"] if q != qubit_id]
                neighbours = [*neighbours, new_neighbours]

    return neighbours
