import json
import os

def read_meta(meta : str) -> dict:
    """quich read information stored in meta.json"""
    assert os.path.exists(meta)
    with open(meta, "r", encoding="utf-8") as fp:
        data = json.load(fp=fp)
    return data