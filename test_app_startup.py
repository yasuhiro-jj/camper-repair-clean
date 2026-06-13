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


def passthrough_cache(func=None, **_kwargs):
    if func is None:
        return lambda wrapped: wrapped
    return func


def install_dependency_stubs():
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = SessionState()
    streamlit.secrets = {}
    streamlit.errors = []
    streamlit.infos = []
    streamlit.warnings = []

    streamlit.cache_resource = passthrough_cache
    streamlit.set_page_config = lambda **_kwargs: None
    streamlit.markdown = lambda *_args, **_kwargs: None
    streamlit.error = lambda message: streamlit.errors.append(message)
    streamlit.info = lambda message: streamlit.infos.append(message)
    streamlit.warning = lambda message: streamlit.warnings.append(message)

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.OpenAIEmbeddings = object

    class ChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, _messages):
            return types.SimpleNamespace(content="【対処法】\nテスト回答")

    langchain_openai.ChatOpenAI = ChatOpenAI

    langchain_core = types.ModuleType("langchain_core")
    langchain_core_messages = types.ModuleType("langchain_core.messages")

    class BaseMessage:
        pass

    class HumanMessage(BaseMessage):
        def __init__(self, content):
            self.content = content

    class AIMessage(BaseMessage):
        def __init__(self, content):
            self.content = content

    langchain_core_messages.BaseMessage = BaseMessage
    langchain_core_messages.HumanMessage = HumanMessage
    langchain_core_messages.AIMessage = AIMessage

    langchain_community = types.ModuleType("langchain_community")
    document_loaders = types.ModuleType("langchain_community.document_loaders")

    class EmptyLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self):
            return []

    document_loaders.PyPDFLoader = EmptyLoader
    document_loaders.TextLoader = EmptyLoader

    modules = {
        "streamlit": streamlit,
        "langchain_openai": langchain_openai,
        "langchain_core": langchain_core,
        "langchain_core.messages": langchain_core_messages,
        "langchain_community": langchain_community,
        "langchain_community.document_loaders": document_loaders,
    }
    sys.modules.update(modules)
    return streamlit


class StartupRegressionTests(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("streamlit_app", None)
        sys.modules.pop("config", None)
        self.streamlit = install_dependency_stubs()

    def tearDown(self):
        sys.modules.pop("streamlit_app", None)
        sys.modules.pop("config", None)

    def import_app(self):
        return importlib.import_module("streamlit_app")

    def test_import_does_not_require_local_config_file(self):
        with patch.dict(os.environ, {}, clear=True):
            app = self.import_app()

        self.assertIsNone(app.get_openai_api_key())
        self.assertIn("messages", self.streamlit.session_state)

    def test_initialize_database_returns_empty_when_no_knowledge_files_exist(self):
        app = self.import_app()

        with patch.object(app.glob, "glob", return_value=[]):
            self.assertEqual([], app.initialize_database())

    def test_missing_model_configuration_does_not_call_none_invoke(self):
        app = self.import_app()
        self.streamlit.session_state.messages = [
            {"role": "user", "content": "バッテリーが上がりました"}
        ]

        with patch.object(app, "initialize_database", return_value=[]), patch.object(app, "build_workflow", return_value=None):
            app.generate_ai_response("バッテリーが上がりました")

        self.assertEqual("assistant", self.streamlit.session_state.messages[-1]["role"])
        self.assertIn("AI回答を生成できません", self.streamlit.session_state.messages[-1]["content"])
        self.assertTrue(self.streamlit.warnings)


if __name__ == "__main__":
    unittest.main()
