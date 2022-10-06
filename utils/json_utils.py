# This code is part of Tergite
#
# (C) Copyright David Wahlstedt 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from pathlib import Path
from typing import List, Dict, Any
import json
import ijson

# Note: if the JSON file is big, this may need to be changed. Perhaps
# the "bigjson" library could help, which reads the file as a stream
# and uses only what's needed, instead of loading the whole file into
# memory
def get_items_from_json(file_path: Path, keys: List[str]) -> Dict[str, Any]:

    with open(file_path, "rb") as f:
        contents = json.load(f)

    items = [(key, contents[key]) for key in keys if key in contents]

    return dict(items)
