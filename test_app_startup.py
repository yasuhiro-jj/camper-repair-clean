import importlib
import os
import sys
import types
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent


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


class ExplodingLoader:
    calls = []

    def __init__(self, path, *args, **kwargs):
        self.calls.append(path)

    def load(self):
        raise AssertionError("loader should not be used when no files are discovered")


class StartupRegressionTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self):
        os.environ.pop("OPENAI_API_KEY", None)
        sys.modules.pop("streamlit_app", None)

    def import_app(self):
        sys.modules.pop("streamlit_app", None)
        sys.modules.pop("config", None)

        st = types.ModuleType("streamlit")
        st.session_state = SessionState()
        st.secrets = {}
        st.set_page_config = lambda *args, **kwargs: None
        st.markdown = lambda *args, **kwargs: None
        st.error = lambda *args, **kwargs: None
        st.info = lambda *args, **kwargs: None
        st.cache_resource = lambda func: func
        sys.modules["streamlit"] = st

        langchain_openai = types.ModuleType("langchain_openai")
        langchain_openai.ChatOpenAI = FakeChatOpenAI
        langchain_openai.OpenAIEmbeddings = object
        sys.modules["langchain_openai"] = langchain_openai

        langchain_core = types.ModuleType("langchain_core")
        messages = types.ModuleType("langchain_core.messages")
        messages.BaseMessage = object
        messages.HumanMessage = lambda content: ("human", content)
        messages.AIMessage = lambda content: ("ai", content)
        sys.modules["langchain_core"] = langchain_core
        sys.modules["langchain_core.messages"] = messages

        loaders = types.ModuleType("langchain_community.document_loaders")
        ExplodingLoader.calls = []
        loaders.PyPDFLoader = ExplodingLoader
        loaders.TextLoader = ExplodingLoader
        sys.modules["langchain_community"] = types.ModuleType("langchain_community")
        sys.modules["langchain_community.document_loaders"] = loaders

        sys.path.insert(0, str(WORKSPACE))
        try:
            return importlib.import_module("streamlit_app"), st
        finally:
            try:
                sys.path.remove(str(WORKSPACE))
            except ValueError:
                pass

    def test_import_does_not_require_local_config_module(self):
        app, _ = self.import_app()

        self.assertTrue(hasattr(app, "initialize_model"))

    def test_model_uses_environment_api_key(self):
        app, _ = self.import_app()
        os.environ["OPENAI_API_KEY"] = "env-key"

        model = app.initialize_model()

        self.assertEqual(model.kwargs["api_key"], "env-key")

    def test_model_uses_streamlit_secrets_api_key(self):
        app, st = self.import_app()
        st.secrets["OPENAI_API_KEY"] = "secret-key"

        model = app.initialize_model()

        self.assertEqual(model.kwargs["api_key"], "secret-key")

    def test_database_initialization_allows_missing_knowledge_files(self):
        app, _ = self.import_app()
        app.glob.glob = lambda pattern: []

        documents = app.initialize_database()

        self.assertEqual(documents, [])
        self.assertEqual(ExplodingLoader.calls, [])


if __name__ == "__main__":
    unittest.main()
