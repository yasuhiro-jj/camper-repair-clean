import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class StreamlitStub(types.ModuleType):
    def __init__(self, secrets=None):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = secrets or {}
        self.errors = []
        self.infos = []

    def cache_resource(self, func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

    def set_page_config(self, **_kwargs):
        return None

    def markdown(self, *_args, **_kwargs):
        return None

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.infos.append(message)


def make_fake_modules(secrets=None):
    streamlit = StreamlitStub(secrets=secrets)

    langchain_openai = types.ModuleType("langchain_openai")

    class FakeChatOpenAI:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

    langchain_openai.ChatOpenAI = FakeChatOpenAI
    langchain_openai.OpenAIEmbeddings = object

    messages = types.ModuleType("langchain_core.messages")

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    messages.BaseMessage = FakeMessage
    messages.HumanMessage = FakeMessage
    messages.AIMessage = FakeMessage

    langchain_core = types.ModuleType("langchain_core")
    langchain_community = types.ModuleType("langchain_community")
    document_loaders = types.ModuleType("langchain_community.document_loaders")

    class FakeLoader:
        def __init__(self, path, *args, **kwargs):
            self.path = path

        def load(self):
            raise AssertionError("loader should not be called when no files exist")

    document_loaders.PyPDFLoader = FakeLoader
    document_loaders.TextLoader = FakeLoader

    return {
        "streamlit": streamlit,
        "langchain_openai": langchain_openai,
        "langchain_core": langchain_core,
        "langchain_core.messages": messages,
        "langchain_community": langchain_community,
        "langchain_community.document_loaders": document_loaders,
    }, streamlit, FakeChatOpenAI


def load_app_module(secrets=None):
    fake_modules, streamlit, fake_chat = make_fake_modules(secrets=secrets)
    module_name = "streamlit_app_under_test"
    sys.modules.pop(module_name, None)

    with patch.dict(sys.modules, fake_modules):
        spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return module, streamlit, fake_chat


class StreamlitAppStartupTest(unittest.TestCase):
    def test_import_does_not_require_untracked_config_module(self):
        module, _streamlit, _fake_chat = load_app_module()

        self.assertFalse(hasattr(module, "config"))

    def test_initialize_database_returns_empty_when_no_documents_exist(self):
        module, _streamlit, _fake_chat = load_app_module()
        module.glob.glob = lambda _pattern: []

        self.assertEqual(module.initialize_database(), [])

    def test_initialize_model_reads_api_key_from_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            module, _streamlit, fake_chat = load_app_module()

            model = module.initialize_model()

        self.assertIs(model, fake_chat.instances[-1])
        self.assertEqual(fake_chat.instances[-1].kwargs["api_key"], "sk-test")

    def test_initialize_model_reads_api_key_from_streamlit_secrets(self):
        with patch.dict(os.environ, {}, clear=True):
            module, _streamlit, fake_chat = load_app_module(
                secrets={"OPENAI_API_KEY": "sk-secret"}
            )

            model = module.initialize_model()

        self.assertIs(model, fake_chat.instances[-1])
        self.assertEqual(fake_chat.instances[-1].kwargs["api_key"], "sk-secret")

    def test_initialize_model_returns_none_when_api_key_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            module, streamlit, _fake_chat = load_app_module()

            model = module.initialize_model()

        self.assertIsNone(model)
        self.assertTrue(streamlit.errors)
        self.assertTrue(streamlit.infos)


if __name__ == "__main__":
    unittest.main()
