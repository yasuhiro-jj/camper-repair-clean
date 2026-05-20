import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ExplodingPdfLoader:
    def __init__(self, path):
        raise AssertionError(f"unexpected PDF load: {path}")


class ExplodingTextLoader:
    def __init__(self, path, encoding=None):
        raise AssertionError(f"unexpected text load: {path}")


def import_app(secrets=None):
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = SessionState()
    streamlit.secrets = secrets if secrets is not None else {}
    streamlit.set_page_config = mock.Mock()
    streamlit.markdown = mock.Mock()
    streamlit.error = mock.Mock()
    streamlit.info = mock.Mock()
    streamlit.cache_resource = lambda func: func

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = FakeChatOpenAI
    langchain_openai.OpenAIEmbeddings = object

    messages = types.ModuleType("langchain_core.messages")
    messages.BaseMessage = object
    messages.HumanMessage = lambda content: ("human", content)
    messages.AIMessage = lambda content: ("ai", content)

    langchain_core = types.ModuleType("langchain_core")
    langchain_core.messages = messages

    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.PyPDFLoader = ExplodingPdfLoader
    loaders.TextLoader = ExplodingTextLoader

    langchain_community = types.ModuleType("langchain_community")
    langchain_community.document_loaders = loaders

    module_path = Path(__file__).with_name("streamlit_app.py")
    spec = importlib.util.spec_from_file_location("streamlit_app_under_test", module_path)
    module = importlib.util.module_from_spec(spec)

    injected_modules = {
        "streamlit": streamlit,
        "langchain_openai": langchain_openai,
        "langchain_core": langchain_core,
        "langchain_core.messages": messages,
        "langchain_community": langchain_community,
        "langchain_community.document_loaders": loaders,
    }

    with mock.patch.dict(sys.modules, injected_modules):
        spec.loader.exec_module(module)

    return module


class StartupConfigurationTests(unittest.TestCase):
    def test_import_does_not_require_local_config_module(self):
        with mock.patch.dict(sys.modules, {"config": None}):
            import_app()

    def test_openai_api_key_comes_from_environment_or_streamlit_secrets(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}):
            app = import_app(secrets={"OPENAI_API_KEY": "from-secrets"})
            self.assertEqual(app.get_openai_api_key(), "from-env")

        with mock.patch.dict(os.environ, {}, clear=True):
            app = import_app(secrets={"OPENAI_API_KEY": "from-secrets"})
            self.assertEqual(app.get_openai_api_key(), "from-secrets")

    def test_initialize_database_returns_empty_when_no_documents_are_bundled(self):
        app = import_app()

        self.assertEqual(app.initialize_database(), [])


if __name__ == "__main__":
    unittest.main()
