import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DummyMessage:
    def __init__(self, content):
        self.content = content


class ExplodingLoader:
    def __init__(self, path, *args, **kwargs):
        self.path = path

    def load(self):
        raise AssertionError(f"unexpected loader call for {self.path}")


def make_streamlit_module():
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = SessionState()
    streamlit.secrets = {}
    streamlit.set_page_config = lambda *args, **kwargs: None
    streamlit.markdown = lambda *args, **kwargs: None
    streamlit.error = lambda *args, **kwargs: None
    streamlit.info = lambda *args, **kwargs: None
    streamlit.warning = lambda *args, **kwargs: None
    streamlit.cache_resource = lambda func=None, **kwargs: func if func else lambda wrapped: wrapped
    return streamlit


def make_langchain_modules():
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = lambda *args, **kwargs: object()
    langchain_openai.OpenAIEmbeddings = lambda *args, **kwargs: object()

    messages = types.ModuleType("langchain_core.messages")
    messages.BaseMessage = DummyMessage
    messages.HumanMessage = DummyMessage
    messages.AIMessage = DummyMessage

    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.PyPDFLoader = ExplodingLoader
    loaders.TextLoader = ExplodingLoader

    return {
        "langchain_openai": langchain_openai,
        "langchain_core.messages": messages,
        "langchain_community.document_loaders": loaders,
    }


class StartupRegressionTests(unittest.TestCase):
    def setUp(self):
        self.original_modules = {}
        test_modules = {"streamlit": make_streamlit_module()}
        test_modules.update(make_langchain_modules())

        for name, module in test_modules.items():
            self.original_modules[name] = sys.modules.get(name)
            sys.modules[name] = module

        self.original_modules["streamlit_app"] = sys.modules.pop("streamlit_app", None)
        self.original_modules["config"] = sys.modules.pop("config", None)

    def tearDown(self):
        for name, module in self.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def import_app(self):
        return importlib.import_module("streamlit_app")

    def test_import_succeeds_without_local_config_module(self):
        app = self.import_app()

        self.assertTrue(callable(app.get_openai_api_key))

    def test_openai_key_can_come_from_environment(self):
        app = self.import_app()

        with patch.dict(os.environ, {"OPENAI_API_KEY": " sk-test "}, clear=False):
            self.assertEqual(app.get_openai_api_key(), "sk-test")

    def test_initialize_database_returns_empty_when_no_documents_are_bundled(self):
        app = self.import_app()

        with patch.object(app.glob, "glob", return_value=[]):
            self.assertEqual(app.initialize_database(), [])

    def test_missing_api_key_adds_assistant_message_without_invoking_model(self):
        app = self.import_app()
        app.st.session_state.messages = [{"role": "user", "content": "question"}]

        with patch.object(app, "initialize_database", return_value=[]), patch.object(app, "build_workflow", return_value=None):
            app.generate_ai_response("question")

        self.assertEqual(app.st.session_state.messages[-1]["role"], "assistant")
        self.assertIn("OpenAI", app.st.session_state.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
