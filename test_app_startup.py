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


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = {}
        self.messages = []

    def set_page_config(self, **kwargs):
        self.page_config = kwargs

    def markdown(self, message, *args, **kwargs):
        self.messages.append(("markdown", message))

    def error(self, message):
        self.messages.append(("error", message))

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def cache_resource(self, func):
        return func


class FakeChatOpenAI:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.invoke_called = False
        FakeChatOpenAI.calls.append(kwargs)

    def invoke(self, messages):
        self.invoke_called = True
        return types.SimpleNamespace(content="answer")


def install_dependency_stubs():
    fake_streamlit = FakeStreamlit()
    sys.modules["streamlit"] = fake_streamlit

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = FakeChatOpenAI
    langchain_openai.OpenAIEmbeddings = object
    sys.modules["langchain_openai"] = langchain_openai

    messages_module = types.ModuleType("langchain_core.messages")
    messages_module.BaseMessage = object
    messages_module.HumanMessage = lambda content: ("human", content)
    messages_module.AIMessage = lambda content: ("ai", content)
    sys.modules["langchain_core"] = types.ModuleType("langchain_core")
    sys.modules["langchain_core.messages"] = messages_module

    document_loaders = types.ModuleType("langchain_community.document_loaders")

    class LoaderShouldNotRun:
        def __init__(self, *args, **kwargs):
            raise AssertionError("missing bundled docs should not load fallback files")

    document_loaders.PyPDFLoader = LoaderShouldNotRun
    document_loaders.TextLoader = LoaderShouldNotRun
    sys.modules["langchain_community"] = types.ModuleType("langchain_community")
    sys.modules["langchain_community.document_loaders"] = document_loaders

    return fake_streamlit


class StartupRegressionTest(unittest.TestCase):
    def setUp(self):
        self.fake_streamlit = install_dependency_stubs()
        FakeChatOpenAI.calls = []
        sys.modules.pop("streamlit_app", None)
        sys.modules.pop("config", None)

    def import_app(self):
        return importlib.import_module("streamlit_app")

    def test_import_does_not_require_local_config_file(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            module = self.import_app()

        self.assertTrue(hasattr(module, "get_openai_api_key"))
        self.assertNotIn("config", sys.modules)

    def test_initialize_database_returns_empty_list_without_bundled_docs(self):
        module = self.import_app()
        with mock.patch.object(module.glob, "glob", return_value=[]):
            self.assertEqual(module.initialize_database(), [])

    def test_initialize_model_reads_openai_key_from_environment(self):
        module = self.import_app()
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            model = module.initialize_model()

        self.assertIsInstance(model, FakeChatOpenAI)
        self.assertEqual(FakeChatOpenAI.calls[-1]["api_key"], "env-key")

    def test_generate_ai_response_does_not_invoke_missing_model(self):
        module = self.import_app()
        module.initialize_database = lambda: []
        module.build_workflow = lambda: None
        self.fake_streamlit.session_state.messages = [
            {"role": "user", "content": "battery help"}
        ]

        module.generate_ai_response("battery help")

        self.assertEqual(
            self.fake_streamlit.session_state.messages[-1]["role"],
            "assistant",
        )
        self.assertTrue(
            any(kind == "warning" for kind, _ in self.fake_streamlit.messages)
        )


if __name__ == "__main__":
    unittest.main()
