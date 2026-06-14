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


def install_dependency_stubs():
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = SessionState()
    streamlit.secrets = {}
    streamlit.warnings = []
    streamlit.errors = []
    streamlit.infos = []
    streamlit.set_page_config = lambda **kwargs: None
    streamlit.markdown = lambda *args, **kwargs: None
    streamlit.error = lambda message: streamlit.errors.append(message)
    streamlit.info = lambda message: streamlit.infos.append(message)
    streamlit.warning = lambda message: streamlit.warnings.append(message)
    streamlit.cache_resource = lambda func=None, **kwargs: func if func else (lambda wrapped: wrapped)

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = lambda **kwargs: types.SimpleNamespace(kwargs=kwargs)
    langchain_openai.OpenAIEmbeddings = object

    messages = types.ModuleType("langchain_core.messages")

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class AIMessage(HumanMessage):
        pass

    messages.BaseMessage = object
    messages.HumanMessage = HumanMessage
    messages.AIMessage = AIMessage

    document_loaders = types.ModuleType("langchain_community.document_loaders")

    class Loader:
        def __init__(self, path, *args, **kwargs):
            self.path = path

        def load(self):
            return []

    document_loaders.PyPDFLoader = Loader
    document_loaders.TextLoader = Loader

    stubs = {
        "streamlit": streamlit,
        "langchain_openai": langchain_openai,
        "langchain_core": types.ModuleType("langchain_core"),
        "langchain_core.messages": messages,
        "langchain_community": types.ModuleType("langchain_community"),
        "langchain_community.document_loaders": document_loaders,
    }

    patcher = patch.dict(sys.modules, stubs)
    patcher.start()
    return patcher, streamlit


class StartupRegressionTests(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("streamlit_app", None)
        sys.modules.pop("config", None)
        os.environ.pop("OPENAI_API_KEY", None)
        self.patcher, self.streamlit = install_dependency_stubs()

    def tearDown(self):
        sys.modules.pop("streamlit_app", None)
        sys.modules.pop("config", None)
        os.environ.pop("OPENAI_API_KEY", None)
        self.patcher.stop()

    def import_app(self):
        return importlib.import_module("streamlit_app")

    def test_app_imports_without_local_config_file(self):
        app = self.import_app()

        self.assertEqual(app.get_openai_api_key(), "")

    def test_openai_api_key_can_come_from_environment(self):
        os.environ["OPENAI_API_KEY"] = "env-key"
        app = self.import_app()

        self.assertEqual(app.get_openai_api_key(), "env-key")

    def test_initialize_database_returns_empty_when_no_documents_are_bundled(self):
        app = self.import_app()

        with patch.object(app.glob, "glob", return_value=[]):
            self.assertEqual(app.initialize_database(), [])

    def test_missing_model_configuration_does_not_invoke_none(self):
        app = self.import_app()
        app.initialize_database = lambda: []
        app.build_workflow = lambda: None
        self.streamlit.session_state.messages = [{"role": "user", "content": "battery help"}]

        app.generate_ai_response("battery help")

        self.assertEqual(self.streamlit.session_state.messages[-1]["role"], "assistant")
        self.assertIn("OpenAI", self.streamlit.session_state.messages[-1]["content"])
        self.assertTrue(self.streamlit.warnings)


if __name__ == "__main__":
    unittest.main()
