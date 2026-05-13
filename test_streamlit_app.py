import importlib
import os
import sys
import types
import unittest
from unittest import mock


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class ChatOpenAIStub:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class LoaderStub:
    def __init__(self, *args, **kwargs):
        pass

    def load(self):
        raise AssertionError("missing fallback files must not be loaded")


class MessageStub:
    def __init__(self, content=None, **kwargs):
        self.content = content


def install_dependency_stubs(secrets=None):
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = SessionState()
    streamlit.secrets = secrets if secrets is not None else {}
    streamlit.set_page_config = lambda **kwargs: None
    streamlit.markdown = lambda *args, **kwargs: None
    streamlit.error = lambda *args, **kwargs: None
    streamlit.info = lambda *args, **kwargs: None

    def cache_resource(func=None, **kwargs):
        def decorator(inner):
            return inner

        return decorator(func) if func is not None else decorator

    streamlit.cache_resource = cache_resource
    sys.modules["streamlit"] = streamlit

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = ChatOpenAIStub
    langchain_openai.OpenAIEmbeddings = object
    sys.modules["langchain_openai"] = langchain_openai

    langchain_core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")
    messages.BaseMessage = MessageStub
    messages.HumanMessage = MessageStub
    messages.AIMessage = MessageStub
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages

    langchain_community = types.ModuleType("langchain_community")
    document_loaders = types.ModuleType("langchain_community.document_loaders")
    document_loaders.PyPDFLoader = LoaderStub
    document_loaders.TextLoader = LoaderStub
    sys.modules["langchain_community"] = langchain_community
    sys.modules["langchain_community.document_loaders"] = document_loaders


def import_app(secrets=None):
    install_dependency_stubs(secrets=secrets)
    sys.modules.pop("streamlit_app", None)
    sys.modules.pop("config", None)
    return importlib.import_module("streamlit_app")


class StreamlitAppConfigTests(unittest.TestCase):
    def test_import_does_not_require_local_config_module(self):
        app = import_app()

        self.assertTrue(hasattr(app, "initialize_model"))

    def test_openai_api_key_comes_from_environment(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            app = import_app()

            self.assertEqual(app.get_openai_api_key(), "env-key")

    def test_openai_api_key_can_come_from_streamlit_secrets(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            app = import_app(secrets={"OPENAI_API_KEY": "secret-key"})

            self.assertEqual(app.get_openai_api_key(), "secret-key")

    def test_initialize_database_returns_empty_when_no_documents_load(self):
        app = import_app()

        self.assertEqual(app.initialize_database(), [])


if __name__ == "__main__":
    unittest.main()
