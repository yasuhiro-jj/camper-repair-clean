import os
from collections.abc import Mapping

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - python-dotenv is optional at runtime.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _coerce_secret_value(value):
    """Return a stripped secret value, or None for missing/empty values."""
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def _secret_get(secrets, key):
    try:
        return secrets[key]
    except Exception:
        return None


def _nested_secret_get(secrets, section, key):
    section_value = _secret_get(secrets, section)
    if section_value is None:
        return None

    if isinstance(section_value, Mapping):
        return section_value.get(key)

    try:
        return section_value[key]
    except Exception:
        return None


def get_openai_api_key(secrets=None, environ=None):
    """Resolve the OpenAI API key from env vars or Streamlit Secrets."""
    environ = os.environ if environ is None else environ

    for key in ("OPENAI_API_KEY", "openai_api_key"):
        value = _coerce_secret_value(environ.get(key))
        if value:
            return value

    if secrets is None:
        return None

    secret_candidates = (
        _secret_get(secrets, "OPENAI_API_KEY"),
        _secret_get(secrets, "openai_api_key"),
        _nested_secret_get(secrets, "openai", "api_key"),
        _nested_secret_get(secrets, "openai", "OPENAI_API_KEY"),
    )
    for value in secret_candidates:
        value = _coerce_secret_value(value)
        if value:
            return value

    return None
