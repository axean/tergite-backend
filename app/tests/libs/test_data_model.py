transmon_list = {
    "q0": {"frequency": 4.8, "anharmonicity": -0.17},
    "q1": {"frequency": 4.225, "anharmonicity": -0.18},
    "q2": {"frequency": 4.35, "anharmonicity": -0.18},
}
coupler_list = {
    "c0": {"frequency": 7.8, "anharmonicity": -0.12},
    "c1": {"frequency": 8.0, "anharmonicity": -0.12},
}
coupling_map = {
    ("q0", "c0"): {"strength": 0.07},
    ("q0", "c1"): {"strength": 0.07},
    ("q1", "c0"): {"strength": 0.07},
    ("q2", "c1"): {"strength": 0.07},
}
