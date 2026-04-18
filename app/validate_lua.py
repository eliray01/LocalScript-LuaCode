import json
import os
import subprocess
import tempfile
from pathlib import Path


def _is_trivial_one_line_return(code: str) -> bool:
    stripped = code.strip()
    if "\n" in stripped or "\r" in stripped:
        return False
    if not stripped.startswith("return "):
        return False
    rest = stripped[len("return ") :].strip()
    if not rest:
        return False
    if ";" in rest:
        return False
    return True


def validate_lua_syntax(code: str) -> tuple[bool, str]:
    """Return (ok, error_message). Uses luac -p when available."""
    code = code.strip()
    if not code:
        return False, "empty code"

    fd, path = tempfile.mkstemp(suffix=".lua")
    os.close(fd)
    tmp = Path(path)
    try:
        tmp.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            ["luac", "-p", str(tmp)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        return False, err
    except FileNotFoundError:
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "luac timeout"
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def validate_lowcode_bindings_json(raw: str) -> tuple[bool, str, dict[str, str]]:
    """
    Task format: a JSON object with one or more keys.
    Each key is an output variable name; each value is JsonString: lua{ <all lua> }lua
    """
    raw = raw.strip()
    if not raw:
        return False, "empty output", {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}", {}

    if not isinstance(data, dict):
        return False, "root JSON value must be an object", {}

    if len(data) < 1:
        return False, "JSON object must contain at least one top-level key", {}

    out: dict[str, str] = {}
    for key, val in data.items():
        if not isinstance(key, str) or not key.strip():
            return False, f"invalid key: {key!r}", {}
        if not isinstance(val, str):
            return False, f"value for {key!r} must be a string (JsonString)", {}
        v = val.strip()
        if not (v.startswith("lua{") and v.endswith("}lua")):
            return (
                False,
                f"value for {key!r} must start with 'lua{{' and end with '}}lua'",
                {},
            )
        inner = v[4:-4]
        if "\n" not in inner and not _is_trivial_one_line_return(inner):
            return (
                False,
                f"value for {key!r} must be multiline Lua, or a trivial one-line 'return ...' expression",
                {},
            )
        ok, err = validate_lua_syntax(inner)
        if not ok:
            return False, f"lua syntax in binding {key!r}: {err}", {}
        out[key] = val

    return True, "", out
