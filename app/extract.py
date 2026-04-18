import re


def extract_json_object(text: str) -> str:
    """Strip optional ```json ... ``` fences; return trimmed text for json.loads."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text
