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
import os
from typing import Union

import numpy as np
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def train_discriminator(
    backend_config_path: Union[str, os.PathLike],
    save_yml_path: Union[str, os.PathLike],
    seed=0,
):
    # We are skipping this test, because the random numbers in the LDA are generated differently on different operating
    # systems, even though we are using the same seed. So, mocking does not work.

    # Read in the qubits and center values from file this could be the general backend configuration
    np.random.seed(seed)
    backend_config = yaml.load(open(backend_config_path, "r"), Loader=yaml.FullLoader)
    i_q_mapping = backend_config["simulator"]["i_q_mapping"]
    cov_matrix = backend_config["simulator"]["cov_matrix"]
    # generate measurement results for two blobs
    sample = np.random.choice([0, 1], size=1000, p=[0.5, 0.5])
    # This can be done with the same function as in the executor
    discriminators = {}
    for qubit in backend_config["qubit_ids"]:
        idx = int(qubit.strip("q"))
        i_q_values = np.array(
            [
                np.random.multivariate_normal(
                    i_q_mapping[idx][s_], np.array(cov_matrix), 1
                )[0]
                for s_ in sample
            ]
        )
        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(i_q_values, sample)

        discriminators[qubit] = {
            "intercept": float(lda_model.intercept_),
            "coef_0": float(lda_model.coef_[0][0]),
            "coef_1": float(lda_model.coef_[0][1]),
        }
    discriminator_config = {"discriminators": {"lda": discriminators}}
    print("trained discriminator")
    with open(save_yml_path, "w") as f:
        yaml.dump(discriminator_config, open(save_yml_path, "w"))
        f.close()
    return discriminator_config


if __name__ == "__main__":
    BACKEND_CONFIG_FILE = "PATH_TO_THE_BACKEND_CONFIGURATION"
    SAVE_YML_PATH = "PATH_TO_THE OUTPUT_FILE"
    train_discriminator(BACKEND_CONFIG_FILE, SAVE_YML_PATH)
