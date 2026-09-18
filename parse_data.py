import json
import os


def load_items(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", [])


def get_unclaimed_items(items):
    return [item for item in items if item.get("status") == "unclaimed"]


def save_result(result, filename):
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)