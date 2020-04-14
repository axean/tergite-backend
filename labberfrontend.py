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


import time


class LabberFrontend:
    @classmethod
    def addJob(cls, date, value):
        time.sleep(10)
        print(f"T1 {value} measured at {date}")

        return None

    def iJob(self, date, value):
        time.sleep(10)
        print(f"T1 {value} measured at {date}")

        return None


def addJob(date, value):
    time.sleep(10)
    print(f"T1 {value} measured at {date}")

    return None
