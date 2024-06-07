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

import numpy as np
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Replace this with the path to your backend configuration
BACKEND_CONFIG_FILE = (
    "/Users/stefanhi/repos/tergite-bcc/app/tests/fixtures/simulator_backend.yml"
)


def train_discriminator():
    # Read in the qubits and center values from file this could be the general backend configuration
    backend_config = yaml.load(open(BACKEND_CONFIG_FILE, "r"), Loader=yaml.FullLoader)
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
    yaml.dump(discriminator_config, open("discriminator_config.yml", "w"))


if __name__ == "__main__":
    train_discriminator()
