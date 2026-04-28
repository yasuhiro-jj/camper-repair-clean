import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class SessionState(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = {}
        self.warnings = []

    def cache_resource(self, func):
        return func

    def set_page_config(self, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def warning(self, message):
        self.warnings.append(message)

    def error(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None


class FailingLoader:
    def __init__(self, path, *args, **kwargs):
        raise AssertionError(f"loader should not be called for missing file: {path}")


def install_dependency_stubs():
    fake_streamlit = FakeStreamlit()
    sys.modules["streamlit"] = fake_streamlit

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = object
    langchain_openai.OpenAIEmbeddings = object
    sys.modules["langchain_openai"] = langchain_openai

    messages = types.ModuleType("langchain_core.messages")
    messages.BaseMessage = object
    messages.HumanMessage = lambda content: ("human", content)
    messages.AIMessage = lambda content: ("ai", content)
    sys.modules["langchain_core"] = types.ModuleType("langchain_core")
    sys.modules["langchain_core.messages"] = messages

    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.PyPDFLoader = FailingLoader
    loaders.TextLoader = FailingLoader
    sys.modules["langchain_community"] = types.ModuleType("langchain_community")
    sys.modules["langchain_community.document_loaders"] = loaders
    return fake_streamlit


class StartupResilienceTest(unittest.TestCase):
    def setUp(self):
        self.fake_streamlit = install_dependency_stubs()
        sys.modules.pop("config", None)
        sys.modules.pop("streamlit_app", None)

    def tearDown(self):
        sys.modules.pop("streamlit_app", None)

    def test_import_without_config_module(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            module = importlib.import_module("streamlit_app")

        self.assertEqual(module.get_openai_api_key(), "")

    def test_initialize_database_without_documents_returns_empty_list(self):
        module = importlib.import_module("streamlit_app")

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_app = Path(temp_dir) / "streamlit_app.py"
            fake_app.write_text("# test app\n", encoding="utf-8")
            module.__file__ = str(fake_app)

            documents = module.initialize_database()

        self.assertEqual(documents, [])
        self.assertTrue(self.fake_streamlit.warnings)


if __name__ == "__main__":
    unittest.main()
