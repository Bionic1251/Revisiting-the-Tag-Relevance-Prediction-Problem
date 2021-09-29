import json


def save_json(obj, save_path):
    with open(save_path, "w", encoding='utf8') as f:
        json.dump(obj, f, indent=4)


def load_json(file_path):
    with open(file_path, "r",  encoding='utf8') as f:
        data = json.load(f)
        return data
