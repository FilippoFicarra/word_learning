import json
import os


def load_json_dict(file_name: str) -> dict:
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            json_dict = json.load(f)
        return json_dict


def save_json_dict(file_path: str, json_dict: dict) -> None:
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    with open(file_path, "w") as f:
        json.dump(json_dict, f, indent=4)
