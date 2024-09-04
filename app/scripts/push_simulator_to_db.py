from app.libs.quantum_executor.qiskit.backend import FakeOpenPulse1Q

# This code is part of Tergite
#
# (C) Pontus Vikstål, Adilet Tuleouv, Stefan Hill (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from app.services.properties.service import post_mss_backend
from app.utils.validation import validate_schema

if __name__ == "__main__":
    backend = FakeOpenPulse1Q(meas_level=1, meas_return="single")
    backend.train_discriminator()
    backend_json = backend.to_db()

    is_valid_backend = validate_schema(backend_json, "backend_v1")

    if is_valid_backend:
        post_mss_backend(backend_json=backend_json)

    print(is_valid_backend)
