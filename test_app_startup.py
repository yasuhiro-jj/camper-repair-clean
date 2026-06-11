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


class StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = {}
        self.errors = []
        self.infos = []

    def cache_resource(self, func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

    def set_page_config(self, **_kwargs):
        pass

    def markdown(self, *_args, **_kwargs):
        pass

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.infos.append(message)


class ChatOpenAIStub:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class LoaderStub:
    def __init__(self, path, *_args, **_kwargs):
        if path.endswith("キャンピングカー修理マニュアル.pdf"):
            raise AssertionError("missing fallback PDF should not be loaded")
        self.path = path

    def load(self):
        return []


class MessageStub:
    def __init__(self, content):
        self.content = content


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.original_api_key = os.environ.pop("OPENAI_API_KEY", None)
        self._install_dependency_stubs()
        sys.modules.pop("streamlit_app", None)
        sys.modules.pop("config", None)

    def tearDown(self):
        if self.original_api_key is not None:
            os.environ["OPENAI_API_KEY"] = self.original_api_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

        for module_name in [
            "streamlit_app",
            "streamlit",
            "langchain_openai",
            "langchain_core",
            "langchain_core.messages",
            "langchain_community",
            "langchain_community.document_loaders",
        ]:
            sys.modules.pop(module_name, None)

    def _install_dependency_stubs(self):
        streamlit = StreamlitStub()
        sys.modules["streamlit"] = streamlit

        langchain_openai = types.ModuleType("langchain_openai")
        langchain_openai.ChatOpenAI = ChatOpenAIStub
        langchain_openai.OpenAIEmbeddings = object
        sys.modules["langchain_openai"] = langchain_openai

        langchain_core = types.ModuleType("langchain_core")
        langchain_core.__path__ = []
        sys.modules["langchain_core"] = langchain_core

        messages = types.ModuleType("langchain_core.messages")
        messages.BaseMessage = MessageStub
        messages.HumanMessage = MessageStub
        messages.AIMessage = MessageStub
        sys.modules["langchain_core.messages"] = messages
        langchain_core.messages = messages

        langchain_community = types.ModuleType("langchain_community")
        langchain_community.__path__ = []
        sys.modules["langchain_community"] = langchain_community

        loaders = types.ModuleType("langchain_community.document_loaders")
        loaders.PyPDFLoader = LoaderStub
        loaders.TextLoader = LoaderStub
        sys.modules["langchain_community.document_loaders"] = loaders
        langchain_community.document_loaders = loaders

    def test_import_does_not_require_local_config_file(self):
        app = importlib.import_module("streamlit_app")

        self.assertIsNone(app.get_openai_api_key())

    def test_initialize_database_allows_no_bundled_knowledge_files(self):
        app = importlib.import_module("streamlit_app")

        self.assertEqual(app.initialize_database(), [])

    def test_missing_api_key_stops_before_model_invoke(self):
        app = importlib.import_module("streamlit_app")
        app.st.session_state.messages = [{"role": "user", "content": "バッテリーが上がりました"}]

        app.generate_ai_response("バッテリーが上がりました")

        self.assertEqual(app.st.session_state.messages[-1]["role"], "assistant")
        self.assertIn("OpenAI APIキーが設定されていないため", app.st.session_state.messages[-1]["content"])
        self.assertFalse(
            any("'NoneType' object has no attribute 'invoke'" in error for error in app.st.errors)
        )


if __name__ == "__main__":
    unittest.main()
