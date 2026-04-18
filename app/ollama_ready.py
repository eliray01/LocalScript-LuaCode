import json
import os
import time
import urllib.error
import urllib.request


def wait_for_ollama_model() -> None:
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
    tag_url = f"{base}/api/tags"
    deadline = time.monotonic() + float(os.environ.get("OLLAMA_READY_TIMEOUT_SEC", "600"))

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(tag_url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            names = [m.get("name", "") for m in data.get("models", [])]
            if any(model == n or n.startswith(model.split(":")[0]) for n in names):
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            pass
        time.sleep(3)

    raise RuntimeError(
        f"Ollama model {model!r} not found at {tag_url} within timeout. "
        "Ensure `ollama pull` completed."
    )
