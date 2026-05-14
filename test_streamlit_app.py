import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DummyStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = {}
        self.errors = []
        self.infos = []

    def cache_resource(self, func=None, **kwargs):
        def decorator(inner):
            return inner

        if func is None:
            return decorator
        return func

    def set_page_config(self, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.infos.append(message)


class DummyChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyLoader:
    def __init__(self, path, *args, **kwargs):
        self.path = path

    def load(self):
        raise AssertionError(f"unexpected load for {self.path}")


def install_dependency_stubs():
    streamlit = DummyStreamlit()

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = DummyChatOpenAI
    langchain_openai.OpenAIEmbeddings = object

    langchain_core = types.ModuleType("langchain_core")
    langchain_core_messages = types.ModuleType("langchain_core.messages")
    langchain_core_messages.BaseMessage = object
    langchain_core_messages.HumanMessage = lambda content: ("human", content)
    langchain_core_messages.AIMessage = lambda content: ("ai", content)

    langchain_community = types.ModuleType("langchain_community")
    document_loaders = types.ModuleType("langchain_community.document_loaders")
    document_loaders.PyPDFLoader = DummyLoader
    document_loaders.TextLoader = DummyLoader

    return {
        "streamlit": streamlit,
        "langchain_openai": langchain_openai,
        "langchain_core": langchain_core,
        "langchain_core.messages": langchain_core_messages,
        "langchain_community": langchain_community,
        "langchain_community.document_loaders": document_loaders,
    }


class StreamlitAppConfigTests(unittest.TestCase):
    def import_app(self):
        sys.modules.pop("streamlit_app", None)
        sys.modules.pop("config", None)
        with patch.dict(sys.modules, install_dependency_stubs()):
            return importlib.import_module("streamlit_app")

    def test_import_does_not_require_untracked_config_module(self):
        app = self.import_app()

        self.assertTrue(hasattr(app, "initialize_model"))
        self.assertNotIn("config", sys.modules)

    def test_initialize_database_returns_empty_without_documents(self):
        app = self.import_app()

        with tempfile.TemporaryDirectory() as tmpdir:
            app.__file__ = str(Path(tmpdir) / "streamlit_app.py")

            self.assertEqual(app.initialize_database(), [])

    def test_initialize_model_reads_openai_key_from_environment(self):
        app = self.import_app()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            model = app.initialize_model()

        self.assertIsInstance(model, DummyChatOpenAI)
        self.assertEqual(model.kwargs["api_key"], "test-key")


if __name__ == "__main__":
    unittest.main()
