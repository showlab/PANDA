"""Agent utility functions — JSON parsing and helpers."""

import json
import re


def safe_load_json(text: str) -> dict:
    """Safely parse JSON from LLM output, handling markdown fences."""
    try:
        cleaned = re.sub(r"```json|```", "", text).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            return json.loads(cleaned[start : end + 1])
        return json.loads(cleaned)
    except Exception as e:
        print(f"[safe_load_json] Error: {e}")
        return {}


def load_provider_object(text: str, role: str) -> dict:
    """Parse a provider response and require a top-level JSON object."""
    parsed = safe_load_json(text)
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    candidate = cleaned[start : end + 1] if start != -1 and end != -1 else cleaned
    try:
        decoded = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        raise ValueError(f"{role} provider must return a valid JSON object") from None
    if not isinstance(decoded, dict) or not isinstance(parsed, dict):
        raise ValueError(f"{role} provider must return a JSON object")
    return parsed
