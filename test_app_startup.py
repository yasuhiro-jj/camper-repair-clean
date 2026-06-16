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


class StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = {}
        self.messages = []

    def cache_resource(self, func=None, **_kwargs):
        def decorator(inner):
            return inner

        return decorator(func) if func else decorator

    def __getattr__(self, name):
        if name in {"chat_message", "spinner", "container"}:
            return lambda *args, **kwargs: _ContextManager()
        return lambda *args, **kwargs: None


class _ContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ChatOpenAIStub:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.invoke_called = False
        ChatOpenAIStub.calls.append(kwargs)

    def invoke(self, _messages):
        self.invoke_called = True
        return types.SimpleNamespace(content="ok")


class LoaderStub:
    def __init__(self, path, *args, **kwargs):
        self.path = path

    def load(self):
        raise AssertionError("No bundled knowledge files should not trigger fallback loading")


def install_dependency_stubs():
    streamlit_stub = StreamlitStub()

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = ChatOpenAIStub
    langchain_openai.OpenAIEmbeddings = object

    langchain_core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")
    messages.BaseMessage = object
    messages.HumanMessage = lambda content: types.SimpleNamespace(content=content, type="human")
    messages.AIMessage = lambda content: types.SimpleNamespace(content=content, type="ai")

    community = types.ModuleType("langchain_community")
    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.PyPDFLoader = LoaderStub
    loaders.TextLoader = LoaderStub

    sys.modules.update(
        {
            "streamlit": streamlit_stub,
            "langchain_openai": langchain_openai,
            "langchain_core": langchain_core,
            "langchain_core.messages": messages,
            "langchain_community": community,
            "langchain_community.document_loaders": loaders,
        }
    )
    return streamlit_stub


def import_app():
    sys.modules.pop("streamlit_app", None)
    sys.modules.pop("config", None)
    ChatOpenAIStub.calls = []
    streamlit_stub = install_dependency_stubs()
    return importlib.import_module("streamlit_app"), streamlit_stub


class StartupRegressionTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("streamlit_app", None)
        os.environ.pop("OPENAI_API_KEY", None)

    def test_import_does_not_require_local_config_file(self):
        app, _streamlit = import_app()

        self.assertTrue(hasattr(app, "get_openai_api_key"))

    def test_initialize_database_returns_empty_list_without_bundled_files(self):
        app, _streamlit = import_app()

        with mock.patch.object(app.glob, "glob", return_value=[]):
            self.assertEqual(app.initialize_database(), [])

    def test_initialize_model_uses_environment_api_key(self):
        os.environ["OPENAI_API_KEY"] = "sk-test"
        app, _streamlit = import_app()

        model = app.initialize_model()

        self.assertIsInstance(model, ChatOpenAIStub)
        self.assertEqual(ChatOpenAIStub.calls[0]["api_key"], "sk-test")

    def test_generate_ai_response_handles_missing_model_without_invoke(self):
        app, streamlit = import_app()
        streamlit.session_state.messages = [{"role": "user", "content": "バッテリーが上がった"}]
        app.initialize_database = lambda: []
        app.build_workflow = lambda: None

        app.generate_ai_response("バッテリーが上がった")

        self.assertEqual(streamlit.session_state.messages[-1]["role"], "assistant")
        self.assertIn("OPENAI_API_KEY", streamlit.session_state.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
