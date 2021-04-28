# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from setuptools import setup, find_packages

REQUIREMENTS = [
    "aiofiles>=0.5.0",
    "fastapi>=0.61.0",
    "motor>=2.2.0",
    "python-multipart>=0.0.5",
    "redis>=3.5.3",
    "requests>=2.24.0",
    "rq>=1.5.0",
    "uvicorn>=0.11.8",
    "numpy>=1.19.1",
    "PyQt5>=5.14.2",
    "h5py>=2.10.0",
    "scipy>=1.4.1",
    "networkx>=2.5.1",
]

setup(
    name="tergite-bcc",
    author_emails="dobsicek@chalmers.se",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    python_requires=">=3.7",
)
