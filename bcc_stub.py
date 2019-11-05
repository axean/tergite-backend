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

import configparser
import random
from pymongo import MongoClient
from datetime import datetime
from os import path


config = configparser.ConfigParser()
config.read(path.abspath(path.join("config.ini")))
DB_URI = config['DATABASE']['DB_URI']

db = MongoClient(DB_URI)["milestone1"]   # select database
data = db.t1_mon                         # select collection

# ISO 8601 date-time
timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
t1_value  = random.SystemRandom().uniform(40,80)

# NDUV struct
entry = {
    "name"  : "T1",
    "date"  : timestamp,
    "unit"  : "us",
    "value" : t1_value
}

data.insert_one(entry)

