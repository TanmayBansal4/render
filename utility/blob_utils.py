import json
import os
import shutil

BASE_DIR = os.path.join(os.getcwd(), "local_blob_storage")

def _full_path(blob_path: str) -> str:
    return os.path.join(BASE_DIR, blob_path.replace("/", os.sep))

def blob_exists(blob_path: str) -> bool:
    return os.path.exists(_full_path(blob_path))

def read_json_from_blob(blob_path: str):
    path = _full_path(blob_path)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json_to_blob(blob_path: str, data):
    path = _full_path(blob_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def copy_blob(src_blob_path: str, dst_blob_path: str):
    src = _full_path(src_blob_path)
    dst = _full_path(dst_blob_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

def delete_blob(blob_path: str):
    path = _full_path(blob_path)
    if os.path.exists(path):
        os.remove(path)
