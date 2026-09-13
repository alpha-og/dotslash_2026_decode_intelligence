import json
import os
import re
import sys
import urllib.request
import uuid

# --- config ---
BASE = "https://opencode.ai/zen/v1"
MODEL = "big-pickle"
SYSTEM = """You are a coding assistant. For chat reply normally.
When you create/edit code, reply EXACTLY:
FILE: path/to/file.py
```python
<full file content>
```
Ask the user for file path or name if unspecified.
"""


# --- chat ---
def chat(messages, session_id):
    data = json.dumps({"model": MODEL, "messages": messages}).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "opencode/1.19.0",
            "x-opencode-client": "cli",
            "x-opencode-project": "global",
            "x-opencode-session": session_id,
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        j = json.loads(r.read().decode())
    return j["choices"][0]["message"]["content"].strip()


# --- harness ---
def apply_edit(path, content):
    cwd = os.path.abspath(os.getcwd())
    abs_path = os.path.abspath(path)
    if os.path.commonpath([abs_path, cwd]) != cwd:
        print(f"[harness] rejected {path}: outside project folder", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    open(abs_path, "w").write(content)


# --- repl (integrated) ---
messages = [{"role": "system", "content": SYSTEM}]
session_id = str(uuid.uuid4())

if __name__ == "__main__":
    while True:
        try:
            q = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in ("exit", "quit"):
            break
        if not q.strip():
            continue
        messages.append({"role": "user", "content": q})

        a = chat(messages, session_id)
        print(a)
        messages.append({"role": "assistant", "content": a})

        m = re.search(r"FILE:\s*(\S+)\s*```python(.*?)```", a, re.DOTALL)
        if m:
            path, code = m.group(1), m.group(2).strip()
            apply_edit(path, code)
            print(f"[harness] wrote {path} ({len(code)} chars)")
