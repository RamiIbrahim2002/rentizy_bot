import os
import json

STORAGE_DIR = "data/conversations"

os.makedirs(STORAGE_DIR, exist_ok=True)

def save_conversation(user_id, content: str):
    with open(f"{STORAGE_DIR}/{user_id}.txt", "w", encoding="utf-8") as f:
        f.write(content)

def load_conversation(user_id) -> str:
    path = f"{STORAGE_DIR}/{user_id}.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""
