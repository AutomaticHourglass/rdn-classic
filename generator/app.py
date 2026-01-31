#!/usr/bin/env python3
"""
chat_client.py

A tiny Python client that mimics the curl example you posted.

Prerequisites
-------------
  pip install requests          # or use httpx / urllib.request

Usage
-----
  python chat_client.py

"""

import json
import pathlib
from typing import Any, Dict

# ----------------------------------------------------------------------
# 1️⃣  Load the tool‑command mapping (optional – only needed if you
#     want to inspect or use it in this script)
# ----------------------------------------------------------------------
TOOL_CMDS_PATH = pathlib.Path(__file__).parent / "tool_commands.json"

try:
    with TOOL_CMDS_PATH.open("r", encoding="utf-8") as f:
        tool_commands: Dict[str, Any] = json.load(f)
except FileNotFoundError:
    print(f"[WARN] {TOOL_CMDS_PATH} not found – continuing without tool mapping.")
    tool_commands = {}

# ----------------------------------------------------------------------
# 2️⃣  Build the JSON payload that matches the curl example
# ----------------------------------------------------------------------
payload: Dict[str, Any] = {
    "model": "openai/gpt-oss-20b",
    "messages": [
        {
            "role": "system",
            "content": "Always answer in rhymes. Today is Thursday"
        },
        {
            "role": "user",
            "content": "What day is it today?"
        }
    ],
    "temperature": 0.7,
    "max_tokens": -1,   # –1 means “no limit” for this toy server
    "stream": False,
}

headers = {"Content-Type": "application/json"}

# ----------------------------------------------------------------------
# 3️⃣  Send the request with the `requests` library
# ----------------------------------------------------------------------
import requests

try:
    resp = requests.post(
        "http://localhost:8901/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=30,           # seconds – tweak if your server takes longer
    )
except requests.exceptions.RequestException as exc:
    print(f"[ERROR] Failed to reach the server: {exc}")
    exit(1)

# ----------------------------------------------------------------------
# 4️⃣  Handle the response
# ----------------------------------------------------------------------
print(f"Status code: {resp.status_code}")

try:
    data = resp.json()
except json.JSONDecodeError:
    print("[ERROR] Response was not valid JSON:")
    print(resp.text)
    exit(1)

# Pretty‑print the returned data
print("Response JSON:")
print(json.dumps(data, indent=2))
