import importlib
import os
import sys
import types
import unittest


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

    def __exit__(self, exc_type, exc, traceback):
        return False


class StartupRegressionTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("OPENAI_API_KEY", None)
        for module_name in [
            "streamlit_app",
            "streamlit",
            "langchain_openai",
            "langchain_core",
            "langchain_core.messages",
            "langchain_community",
            "langchain_community.document_loaders",
            "langchain_community.document_loaders.pdf",
            "langchain_community.document_loaders.text",
            "config",
        ]:
            sys.modules.pop(module_name, None)

        self.streamlit = self._install_streamlit_stub()
        self._install_langchain_stubs()

    def _install_streamlit_stub(self):
        streamlit = types.ModuleType("streamlit")
        streamlit.session_state = SessionState()
        streamlit.secrets = {}
        streamlit.messages = []

        def cache_resource(func=None, **_kwargs):
            if func is None:
                return lambda wrapped: wrapped
            return func

        def record_message(kind):
            def recorder(message="", *args, **kwargs):
                streamlit.messages.append((kind, message))

            return recorder

        streamlit.cache_resource = cache_resource
        streamlit.set_page_config = lambda *args, **kwargs: None
        streamlit.markdown = record_message("markdown")
        streamlit.error = record_message("error")
        streamlit.info = record_message("info")
        streamlit.warning = record_message("warning")
        streamlit.divider = lambda *args, **kwargs: None
        streamlit.columns = lambda count: [DummyContext() for _ in range(count)]
        streamlit.button = lambda *args, **kwargs: False
        streamlit.rerun = lambda *args, **kwargs: None
        streamlit.chat_input = lambda *args, **kwargs: None
        streamlit.chat_message = lambda *args, **kwargs: DummyContext()
        streamlit.spinner = lambda *args, **kwargs: DummyContext()
        streamlit.container = lambda *args, **kwargs: DummyContext()

        sys.modules["streamlit"] = streamlit
        return streamlit

    def _install_langchain_stubs(self):
        langchain_openai = types.ModuleType("langchain_openai")

        class ChatOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def invoke(self, messages):
                return types.SimpleNamespace(content="stub response")

        langchain_openai.ChatOpenAI = ChatOpenAI
        langchain_openai.OpenAIEmbeddings = object
        sys.modules["langchain_openai"] = langchain_openai

        langchain_core = types.ModuleType("langchain_core")
        messages_module = types.ModuleType("langchain_core.messages")

        class BaseMessage:
            def __init__(self, content=""):
                self.content = content

        class HumanMessage(BaseMessage):
            pass

        class AIMessage(BaseMessage):
            pass

        messages_module.BaseMessage = BaseMessage
        messages_module.HumanMessage = HumanMessage
        messages_module.AIMessage = AIMessage
        sys.modules["langchain_core"] = langchain_core
        sys.modules["langchain_core.messages"] = messages_module

        langchain_community = types.ModuleType("langchain_community")
        loaders_module = types.ModuleType("langchain_community.document_loaders")
        pdf_module = types.ModuleType("langchain_community.document_loaders.pdf")
        text_module = types.ModuleType("langchain_community.document_loaders.text")

        class EmptyLoader:
            def __init__(self, path, *args, **kwargs):
                self.path = path

            def load(self):
                return []

        loaders_module.PyPDFLoader = EmptyLoader
        loaders_module.TextLoader = EmptyLoader
        pdf_module.PyPDFLoader = EmptyLoader
        text_module.TextLoader = EmptyLoader
        sys.modules["langchain_community"] = langchain_community
        sys.modules["langchain_community.document_loaders"] = loaders_module
        sys.modules["langchain_community.document_loaders.pdf"] = pdf_module
        sys.modules["langchain_community.document_loaders.text"] = text_module

    def import_app(self):
        return importlib.import_module("streamlit_app")

    def test_import_without_local_config_uses_environment_key(self):
        os.environ["OPENAI_API_KEY"] = "env-key"

        app = self.import_app()

        self.assertEqual("env-key", app.get_openai_api_key())

    def test_initialize_database_returns_empty_without_bundled_docs(self):
        app = self.import_app()
        app.glob.glob = lambda _pattern: []

        self.assertEqual([], app.initialize_database())

    def test_generate_ai_response_without_model_does_not_invoke_none(self):
        app = self.import_app()
        app.initialize_database = lambda: []
        app.build_workflow = lambda: None
        app.st.session_state.messages = [{"role": "user", "content": "battery help"}]

        app.generate_ai_response("battery help")

        self.assertEqual("assistant", app.st.session_state.messages[-1]["role"])
        self.assertIn("OPENAI_API_KEY", app.st.session_state.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
