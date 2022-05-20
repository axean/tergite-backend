# This code is part of Tergite
#
# (C) Copyright David Wahlstedt 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import argparse

import redis
import utils.redis

if __name__ == "__main__":
    # Set up Redis
    red = redis.Redis(decode_responses=True)

    parser = argparse.ArgumentParser(description="Redis dump utility")
    parser.add_argument("--message", "-m", default="", type=str)
    parser.add_argument("--regex", "-r")
    args = parser.parse_args()

    msg = args.message
    regex = args.regex

    if regex:
        utils.redis.dump_redis(red, msg, regex)
    else:
        utils.redis.dump_redis(red, msg)
