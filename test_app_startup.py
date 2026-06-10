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


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = {}
        self.messages = []

    def cache_resource(self, func=None, **_kwargs):
        def decorator(inner):
            return inner

        return decorator(func) if func is not None else decorator

    def set_page_config(self, **kwargs):
        self.messages.append(("set_page_config", kwargs))

    def markdown(self, content, **_kwargs):
        self.messages.append(("markdown", content))

    def error(self, content):
        self.messages.append(("error", content))

    def info(self, content):
        self.messages.append(("info", content))

    def warning(self, content):
        self.messages.append(("warning", content))


class FakeMessage:
    def __init__(self, content=""):
        self.content = content


class FakeLoader:
    def __init__(self, path, **_kwargs):
        self.path = path

    def load(self):
        raise AssertionError(f"Unexpected fallback load attempted: {self.path}")


def dependency_modules(fake_streamlit):
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = lambda **_kwargs: object()
    langchain_openai.OpenAIEmbeddings = object

    langchain_core = types.ModuleType("langchain_core")
    langchain_core_messages = types.ModuleType("langchain_core.messages")
    langchain_core_messages.BaseMessage = FakeMessage
    langchain_core_messages.HumanMessage = FakeMessage
    langchain_core_messages.AIMessage = FakeMessage

    langchain_community = types.ModuleType("langchain_community")
    document_loaders = types.ModuleType("langchain_community.document_loaders")
    document_loaders.PyPDFLoader = FakeLoader
    document_loaders.TextLoader = FakeLoader

    return {
        "streamlit": fake_streamlit,
        "langchain_openai": langchain_openai,
        "langchain_core": langchain_core,
        "langchain_core.messages": langchain_core_messages,
        "langchain_community": langchain_community,
        "langchain_community.document_loaders": document_loaders,
    }


def import_app(fake_streamlit=None):
    fake_streamlit = fake_streamlit or FakeStreamlit()
    sys.modules.pop("streamlit_app", None)
    sys.modules.pop("config", None)
    with patch.dict(sys.modules, dependency_modules(fake_streamlit)):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            return importlib.import_module("streamlit_app")


class StartupRegressionTests(unittest.TestCase):
    def test_import_does_not_require_local_config_file(self):
        app = import_app()

        self.assertTrue(hasattr(app, "get_openai_api_key"))
        self.assertIsNone(app.get_openai_api_key())

    def test_openai_key_can_come_from_environment_or_streamlit_secrets(self):
        fake_streamlit = FakeStreamlit()
        app = import_app(fake_streamlit)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            self.assertEqual(app.get_openai_api_key(), "env-key")

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            fake_streamlit.secrets = {"OPENAI_API_KEY": "secret-key"}
            self.assertEqual(app.get_openai_api_key(), "secret-key")

    def test_initialize_database_returns_empty_when_no_knowledge_files_exist(self):
        app = import_app()

        with patch.object(app.glob, "glob", return_value=[]):
            self.assertEqual(app.initialize_database(), [])

    def test_generate_ai_response_does_not_invoke_missing_model(self):
        fake_streamlit = FakeStreamlit()
        app = import_app(fake_streamlit)
        fake_streamlit.session_state.messages = [
            {"role": "user", "content": "バッテリーが上がりました"}
        ]

        with patch.object(app, "initialize_database", return_value=[]):
            with patch.object(app, "build_workflow", return_value=None):
                app.generate_ai_response("バッテリーが上がりました")

        self.assertEqual(fake_streamlit.session_state.messages[-1]["role"], "assistant")
        self.assertIn("OPENAI_API_KEY", fake_streamlit.session_state.messages[-1]["content"])
        self.assertIn(("warning", fake_streamlit.session_state.messages[-1]["content"]), fake_streamlit.messages)


if __name__ == "__main__":
    unittest.main()
