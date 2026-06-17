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


class DummyMessage:
    def __init__(self, content):
        self.content = content


class StreamlitAppStartupTests(unittest.TestCase):
    def setUp(self):
        self.original_modules = {}
        for name in [
            "streamlit",
            "langchain_openai",
            "langchain_core",
            "langchain_core.messages",
            "langchain_community",
            "langchain_community.document_loaders",
            "config",
            "streamlit_app",
        ]:
            if name in sys.modules:
                self.original_modules[name] = sys.modules[name]
                del sys.modules[name]

        streamlit = types.ModuleType("streamlit")
        streamlit.session_state = SessionState()
        streamlit.secrets = {}
        streamlit.set_page_config = mock.Mock()
        streamlit.markdown = mock.Mock()
        streamlit.error = mock.Mock()
        streamlit.info = mock.Mock()
        streamlit.cache_resource = lambda func: func
        sys.modules["streamlit"] = streamlit
        self.streamlit = streamlit

        langchain_openai = types.ModuleType("langchain_openai")
        langchain_openai.ChatOpenAI = mock.Mock(return_value=mock.Mock())
        langchain_openai.OpenAIEmbeddings = mock.Mock()
        sys.modules["langchain_openai"] = langchain_openai

        messages = types.ModuleType("langchain_core.messages")
        messages.BaseMessage = DummyMessage
        messages.HumanMessage = DummyMessage
        messages.AIMessage = DummyMessage
        sys.modules["langchain_core"] = types.ModuleType("langchain_core")
        sys.modules["langchain_core.messages"] = messages

        loaders = types.ModuleType("langchain_community.document_loaders")
        loaders.PyPDFLoader = mock.Mock()
        loaders.TextLoader = mock.Mock()
        sys.modules["langchain_community"] = types.ModuleType("langchain_community")
        sys.modules["langchain_community.document_loaders"] = loaders

        os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self):
        for name in [
            "streamlit",
            "langchain_openai",
            "langchain_core",
            "langchain_core.messages",
            "langchain_community",
            "langchain_community.document_loaders",
            "config",
            "streamlit_app",
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(self.original_modules)
        os.environ.pop("OPENAI_API_KEY", None)

    def import_app(self):
        return importlib.import_module("streamlit_app")

    def test_import_does_not_require_local_config_module(self):
        app = self.import_app()

        self.assertIsNone(app.get_openai_api_key())

    def test_openai_api_key_can_come_from_environment(self):
        os.environ["OPENAI_API_KEY"] = "env-key"
        app = self.import_app()

        self.assertEqual(app.get_openai_api_key(), "env-key")

    def test_initialize_database_does_not_load_missing_fallback_pdf(self):
        app = self.import_app()
        app.PyPDFLoader = mock.Mock(side_effect=AssertionError("fallback PDF should not be loaded"))

        with mock.patch.object(app.glob, "glob", return_value=[]):
            self.assertEqual(app.initialize_database(), [])

        app.PyPDFLoader.assert_not_called()

    def test_missing_model_configuration_records_assistant_message(self):
        app = self.import_app()
        self.streamlit.session_state.messages = [{"role": "user", "content": "battery help"}]

        with mock.patch.object(app, "initialize_database", return_value=[]), mock.patch.object(
            app, "build_workflow", return_value=None
        ):
            app.generate_ai_response("battery help")

        self.assertEqual(self.streamlit.session_state.messages[-1]["role"], "assistant")
        self.streamlit.markdown.assert_any_call(
            "OpenAI APIキーが未設定のため、AI回答を生成できません。管理者に設定を確認してください。"
        )


if __name__ == "__main__":
    unittest.main()
