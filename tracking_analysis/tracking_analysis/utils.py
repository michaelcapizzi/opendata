import json
from typing import Dict, Any


def read_in_json(path_to_json: str) -> Dict[Any, Any]:
    with open(path_to_json, "rb") as f:
        json_as_dict = json.load(f)
    return json_as_dict
