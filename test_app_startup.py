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


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _no_op(*args, **kwargs):
    return None


def _install_dependency_stubs():
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = SessionState()
    streamlit.secrets = {}
    streamlit.cache_resource = lambda func=None, **kwargs: (
        func if func is not None else lambda wrapped: wrapped
    )
    streamlit.set_page_config = _no_op
    streamlit.markdown = _no_op
    streamlit.error = _no_op
    streamlit.info = _no_op
    streamlit.divider = _no_op
    streamlit.button = lambda *args, **kwargs: False
    streamlit.rerun = _no_op
    streamlit.columns = lambda *args, **kwargs: [DummyContext(), DummyContext()]
    streamlit.container = lambda *args, **kwargs: DummyContext()
    streamlit.chat_message = lambda *args, **kwargs: DummyContext()
    streamlit.spinner = lambda *args, **kwargs: DummyContext()
    streamlit.chat_input = lambda *args, **kwargs: None

    class FakeChatOpenAI:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.invoked = False
            FakeChatOpenAI.instances.append(self)

        def invoke(self, messages):
            self.invoked = True
            return types.SimpleNamespace(content="response")

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = FakeChatOpenAI
    langchain_openai.OpenAIEmbeddings = object

    messages = types.ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content=""):
            self.content = content

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    messages.BaseMessage = BaseMessage
    messages.HumanMessage = HumanMessage
    messages.AIMessage = AIMessage

    loaders = types.ModuleType("langchain_community.document_loaders")

    class FakeLoader:
        def __init__(self, path, *args, **kwargs):
            self.path = path

        def load(self):
            return []

    loaders.PyPDFLoader = FakeLoader
    loaders.TextLoader = FakeLoader

    sys.modules.update(
        {
            "streamlit": streamlit,
            "langchain_openai": langchain_openai,
            "langchain_core": types.ModuleType("langchain_core"),
            "langchain_core.messages": messages,
            "langchain_community": types.ModuleType("langchain_community"),
            "langchain_community.document_loaders": loaders,
        }
    )
    return streamlit, FakeChatOpenAI


def import_app():
    sys.modules.pop("streamlit_app", None)
    sys.modules.pop("config", None)
    streamlit, fake_chat = _install_dependency_stubs()
    return importlib.import_module("streamlit_app"), streamlit, fake_chat


class StartupRegressionTests(unittest.TestCase):
    def test_import_does_not_require_local_config_file(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            app, streamlit, _ = import_app()

        self.assertTrue(callable(app.get_openai_api_key))
        self.assertEqual(streamlit.session_state.messages, [])

    def test_initialize_database_returns_empty_list_when_no_documents_exist(self):
        app, _, _ = import_app()

        with mock.patch.object(app.glob, "glob", return_value=[]):
            self.assertEqual(app.initialize_database(), [])

    def test_initialize_model_uses_environment_api_key(self):
        app, _, fake_chat = import_app()

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            model = app.initialize_model()

        self.assertIs(model, fake_chat.instances[-1])
        self.assertEqual(model.kwargs["api_key"], "env-key")

    def test_missing_api_key_does_not_invoke_none_model(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            app, streamlit, _ = import_app()

        app.initialize_database = lambda: []
        app.build_workflow = lambda: None
        streamlit.session_state.messages = [{"role": "user", "content": "question"}]

        app.generate_ai_response("question")

        self.assertEqual(streamlit.session_state.messages[-1]["role"], "assistant")
        self.assertIn("OPENAI_API_KEY", streamlit.session_state.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
