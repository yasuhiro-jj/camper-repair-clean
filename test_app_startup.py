import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


APP_PATH = Path(__file__).with_name("streamlit_app.py")


class FakeSessionState(dict):
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
        self.session_state = FakeSessionState()
        self.secrets = {}
        self.errors = []
        self.infos = []

    def cache_resource(self, func):
        return func

    def set_page_config(self, **kwargs):
        self.page_config = kwargs

    def markdown(self, *args, **kwargs):
        return None

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.infos.append(message)


class DummyChatOpenAI:
    last_kwargs = None

    def __init__(self, **kwargs):
        DummyChatOpenAI.last_kwargs = kwargs


class DummyLoader:
    calls = []

    def __init__(self, path, *args, **kwargs):
        self.path = path
        DummyLoader.calls.append(path)

    def load(self):
        return []


class StreamlitAppStartupTests(unittest.TestCase):
    def import_app(self, streamlit_module=None):
        fake_st = streamlit_module or FakeStreamlit()

        modules = {
            "streamlit": fake_st,
            "langchain_openai": types.SimpleNamespace(
                ChatOpenAI=DummyChatOpenAI,
                OpenAIEmbeddings=object,
            ),
            "langchain_core": types.ModuleType("langchain_core"),
            "langchain_core.messages": types.SimpleNamespace(
                BaseMessage=object,
                HumanMessage=lambda content: ("human", content),
                AIMessage=lambda content: ("ai", content),
            ),
            "langchain_community": types.ModuleType("langchain_community"),
            "langchain_community.document_loaders": types.SimpleNamespace(
                PyPDFLoader=DummyLoader,
                TextLoader=DummyLoader,
            ),
        }

        module_name = "streamlit_app_under_test"
        sys.modules.pop(module_name, None)

        with mock.patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
            app = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = app
            spec.loader.exec_module(app)

        return app, fake_st

    def test_app_import_does_not_require_local_config_file(self):
        sys.modules.pop("config", None)

        self.import_app()

    def test_openai_api_key_comes_from_environment(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            app, _ = self.import_app()

        self.assertEqual(app.get_openai_api_key(), "env-key")

    def test_openai_api_key_can_come_from_streamlit_secrets(self):
        fake_st = FakeStreamlit()
        fake_st.secrets["OPENAI_API_KEY"] = "secret-key"

        with mock.patch.dict(os.environ, {}, clear=True):
            app, _ = self.import_app(fake_st)

        self.assertEqual(app.get_openai_api_key(), "secret-key")

    def test_empty_document_set_does_not_load_missing_fallback_pdf(self):
        app, _ = self.import_app()
        DummyLoader.calls = []

        with mock.patch.object(app.glob, "glob", return_value=[]):
            documents = app.initialize_database()

        self.assertEqual(documents, [])
        self.assertEqual(DummyLoader.calls, [])


if __name__ == "__main__":
    unittest.main()
