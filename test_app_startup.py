import importlib
import os
import sys
import types
import unittest
from unittest import mock


class DummySessionState(dict):
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
        self.session_state = DummySessionState()
        self.secrets = {}
        self.messages = []

    def cache_resource(self, func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

    def set_page_config(self, *_args, **_kwargs):
        pass

    def markdown(self, *_args, **_kwargs):
        pass

    def error(self, message, *_args, **_kwargs):
        self.messages.append(("error", message))

    def info(self, message, *_args, **_kwargs):
        self.messages.append(("info", message))

    def warning(self, message, *_args, **_kwargs):
        self.messages.append(("warning", message))


class DummyHumanMessage:
    def __init__(self, content):
        self.content = content


class DummyAIMessage(DummyHumanMessage):
    pass


class StartupRegressionTests(unittest.TestCase):
    def setUp(self):
        self.original_modules = sys.modules.copy()
        self.streamlit = DummyStreamlit()
        self._install_dependency_stubs()
        os.environ.pop("OPENAI_API_KEY", None)
        sys.modules.pop("streamlit_app", None)
        sys.modules.pop("config", None)

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self.original_modules)
        os.environ.pop("OPENAI_API_KEY", None)

    def _install_dependency_stubs(self):
        sys.modules["streamlit"] = self.streamlit

        openai_module = types.ModuleType("langchain_openai")
        openai_module.OpenAIEmbeddings = object

        class DummyChatOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def invoke(self, _messages):
                return types.SimpleNamespace(content="回答")

        openai_module.ChatOpenAI = DummyChatOpenAI
        sys.modules["langchain_openai"] = openai_module

        core_messages_module = types.ModuleType("langchain_core.messages")
        core_messages_module.BaseMessage = object
        core_messages_module.HumanMessage = DummyHumanMessage
        core_messages_module.AIMessage = DummyAIMessage
        sys.modules["langchain_core"] = types.ModuleType("langchain_core")
        sys.modules["langchain_core.messages"] = core_messages_module

        loaders_module = types.ModuleType("langchain_community.document_loaders")

        class FailingPdfLoader:
            def __init__(self, path):
                self.path = path

            def load(self):
                raise FileNotFoundError(self.path)

        class DummyTextLoader(FailingPdfLoader):
            pass

        loaders_module.PyPDFLoader = FailingPdfLoader
        loaders_module.TextLoader = DummyTextLoader
        sys.modules["langchain_community"] = types.ModuleType("langchain_community")
        sys.modules["langchain_community.document_loaders"] = loaders_module

    def import_app(self):
        return importlib.import_module("streamlit_app")

    def test_import_does_not_require_local_config_file(self):
        app = self.import_app()

        self.assertEqual("", app.get_openai_api_key())

    def test_database_initialization_allows_no_bundled_documents(self):
        app = self.import_app()

        with mock.patch.object(app.glob, "glob", return_value=[]):
            self.assertEqual([], app.initialize_database())

    def test_missing_api_key_does_not_call_none_model(self):
        app = self.import_app()
        self.streamlit.session_state.messages = [{"role": "user", "content": "水が出ない"}]

        with mock.patch.object(app, "initialize_database", return_value=[]):
            app.generate_ai_response("水が出ない")

        self.assertEqual("assistant", self.streamlit.session_state.messages[-1]["role"])
        self.assertIn("OPENAI_API_KEY", self.streamlit.session_state.messages[-1]["content"])

    def test_environment_api_key_initializes_model(self):
        os.environ["OPENAI_API_KEY"] = "test-key"
        app = self.import_app()

        model = app.initialize_model()

        self.assertEqual("test-key", model.kwargs["api_key"])


if __name__ == "__main__":
    unittest.main()
