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


class StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = {}
        self.errors = []
        self.infos = []
        self.markdowns = []

    def set_page_config(self, **kwargs):
        self.page_config = kwargs

    def markdown(self, content, *args, **kwargs):
        self.markdowns.append(content)

    def error(self, content, *args, **kwargs):
        self.errors.append(content)

    def info(self, content, *args, **kwargs):
        self.infos.append(content)

    def cache_resource(self, func=None, **kwargs):
        def decorator(wrapped):
            return wrapped

        return decorator(func) if func is not None else decorator


class ChatOpenAIStub:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def invoke(self, messages):
        return types.SimpleNamespace(content="ok")


class MessageStub:
    def __init__(self, content):
        self.content = content


class LoaderStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def load(self):
        return []


class StartupTests(unittest.TestCase):
    MODULE_NAMES = [
        "streamlit_app",
        "streamlit",
        "langchain_openai",
        "langchain_core",
        "langchain_core.messages",
        "langchain_community",
        "langchain_community.document_loaders",
        "config",
    ]

    def setUp(self):
        self.original_modules = {
            name: sys.modules[name] for name in self.MODULE_NAMES if name in sys.modules
        }
        self.addCleanup(self.restore_modules)

    def restore_modules(self):
        for name in self.MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(self.original_modules)

    def import_app(self):
        for name in self.MODULE_NAMES:
            sys.modules.pop(name, None)

        streamlit = StreamlitStub()

        langchain_openai = types.ModuleType("langchain_openai")
        ChatOpenAIStub.instances = []
        langchain_openai.ChatOpenAI = ChatOpenAIStub
        langchain_openai.OpenAIEmbeddings = object

        langchain_core = types.ModuleType("langchain_core")
        messages = types.ModuleType("langchain_core.messages")
        messages.BaseMessage = MessageStub
        messages.HumanMessage = MessageStub
        messages.AIMessage = MessageStub

        langchain_community = types.ModuleType("langchain_community")
        loaders = types.ModuleType("langchain_community.document_loaders")
        loaders.PyPDFLoader = LoaderStub
        loaders.TextLoader = LoaderStub

        sys.modules.update(
            {
                "streamlit": streamlit,
                "langchain_openai": langchain_openai,
                "langchain_core": langchain_core,
                "langchain_core.messages": messages,
                "langchain_community": langchain_community,
                "langchain_community.document_loaders": loaders,
            }
        )

        return importlib.import_module("streamlit_app"), streamlit

    def test_import_does_not_require_local_config_module(self):
        with patch.dict(os.environ, {}, clear=True):
            app, _ = self.import_app()

        self.assertTrue(hasattr(app, "initialize_model"))

    def test_initialize_database_returns_empty_when_no_documents_are_bundled(self):
        app, _ = self.import_app()

        with patch.object(app.glob, "glob", return_value=[]), patch.object(
            app, "PyPDFLoader", side_effect=AssertionError("missing fallback PDF loaded")
        ):
            self.assertEqual(app.initialize_database(), [])

    def test_initialize_model_uses_environment_api_key_without_config(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            app, _ = self.import_app()
            model = app.initialize_model()

        self.assertIs(model, ChatOpenAIStub.instances[-1])
        self.assertEqual(model.kwargs["api_key"], "env-key")

    def test_generate_ai_response_stops_cleanly_when_model_is_unconfigured(self):
        with patch.dict(os.environ, {}, clear=True):
            app, streamlit = self.import_app()
            streamlit.session_state.messages = [{"role": "user", "content": "hello"}]
            app.generate_ai_response("hello")

        self.assertEqual(streamlit.session_state.messages[-1]["role"], "assistant")
        self.assertIn("OPENAI_API_KEY", streamlit.session_state.messages[-1]["content"])
        self.assertFalse(ChatOpenAIStub.instances)


if __name__ == "__main__":
    unittest.main()
