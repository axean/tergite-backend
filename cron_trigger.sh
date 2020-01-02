#!/bin/bash

# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

eval "$(/mnt/extra_ssd/anaconda/anaconda3/bin/conda shell.bash hook)"
conda activate rest-eve

sleep $((RANDOM % (60*59*2)))  # sleep between 1 second upto almost 2 hours

cd /home/dobsicek/bcc_stable
git checkout master
python bcc_stub.py

