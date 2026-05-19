import importlib
import os


def _clean_api_key(value):
    """Return a usable API key string, or None for missing/blank values."""
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def _api_key_from_secrets(secrets):
    if secrets is None:
        return None

    try:
        return _clean_api_key(secrets.get("OPENAI_API_KEY"))
    except AttributeError:
        try:
            return _clean_api_key(secrets["OPENAI_API_KEY"])
        except (KeyError, TypeError):
            return None
    except Exception:
        # Missing Streamlit secrets should not prevent env/config fallbacks.
        try:
            return _clean_api_key(secrets["OPENAI_API_KEY"])
        except Exception:
            return None


def _api_key_from_config_module():
    try:
        config = importlib.import_module("config")
    except ModuleNotFoundError as exc:
        if exc.name != "config":
            raise
        return None

    return _clean_api_key(getattr(config, "OPENAI_API_KEY", None))


def get_openai_api_key(secrets=None, environ=None):
    """Resolve the OpenAI API key without requiring a local config.py file."""
    environ = os.environ if environ is None else environ

    return (
        _clean_api_key(environ.get("OPENAI_API_KEY"))
        or _api_key_from_secrets(secrets)
        or _api_key_from_config_module()
    )
