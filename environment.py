import json
import pickle
from pathlib import Path


def create_folder(path: str):
    folder = Path(path)
    folder.mkdir(exist_ok=True, parents=True)


def create_json_file(data, path: str):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


def load_json_file(path: str) -> dict | list:
    with open(path, "r") as json_file:
        data = json.load(json_file)

    return data


def create_pkl_file(data, output_path: str):
    with open(output_path, 'wb') as file_pi:
        pickle.dump(data, file_pi)


def load_pkl_file(path: str):
    with open(path, "rb") as file_pi:
        data = pickle.load(file_pi)

    return data
