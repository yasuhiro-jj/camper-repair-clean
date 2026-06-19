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
        self.warnings = []
        self.markdowns = []

    def set_page_config(self, **kwargs):
        return None

    def markdown(self, content, **kwargs):
        self.markdowns.append(content)

    def error(self, content):
        self.errors.append(content)

    def info(self, content):
        self.infos.append(content)

    def warning(self, content):
        self.warnings.append(content)

    def cache_resource(self, func):
        return func


class ChatOpenAIStub:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        ChatOpenAIStub.calls.append(kwargs)


class DocumentLoaderStub:
    def __init__(self, path, *args, **kwargs):
        self.path = path

    def load(self):
        raise AssertionError("loader should not be called when no files are bundled")


def install_dependency_stubs():
    st = StreamlitStub()

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = ChatOpenAIStub
    langchain_openai.OpenAIEmbeddings = object

    messages = types.ModuleType("langchain_core.messages")
    messages.BaseMessage = object
    messages.HumanMessage = lambda content: ("human", content)
    messages.AIMessage = lambda content: ("ai", content)

    langchain_core = types.ModuleType("langchain_core")
    langchain_core.messages = messages

    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.PyPDFLoader = DocumentLoaderStub
    loaders.TextLoader = DocumentLoaderStub

    langchain_community = types.ModuleType("langchain_community")
    langchain_community.document_loaders = loaders

    sys.modules.update(
        {
            "streamlit": st,
            "langchain_openai": langchain_openai,
            "langchain_core": langchain_core,
            "langchain_core.messages": messages,
            "langchain_community": langchain_community,
            "langchain_community.document_loaders": loaders,
        }
    )
    return st


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()
        for name in list(sys.modules):
            if name == "streamlit_app" or name.startswith("config"):
                sys.modules.pop(name, None)
        os.environ.pop("OPENAI_API_KEY", None)
        ChatOpenAIStub.calls.clear()
        self.st = install_dependency_stubs()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)
        for name in [
            "streamlit_app",
            "streamlit",
            "langchain_openai",
            "langchain_core",
            "langchain_core.messages",
            "langchain_community",
            "langchain_community.document_loaders",
        ]:
            sys.modules.pop(name, None)

    def import_app(self):
        return importlib.import_module("streamlit_app")

    def test_import_does_not_require_local_config_file(self):
        app = self.import_app()

        self.assertTrue(hasattr(app, "initialize_model"))
        self.assertNotIn("config", sys.modules)

    def test_missing_knowledge_files_returns_empty_documents(self):
        app = self.import_app()

        self.assertEqual(app.initialize_database(), [])

    def test_missing_api_key_does_not_call_model_invoke(self):
        app = self.import_app()

        app.generate_ai_response("バッテリーが上がった")

        self.assertTrue(self.st.warnings)
        self.assertEqual(
            self.st.session_state.messages[-1]["role"],
            "assistant",
        )
        self.assertIn("OPENAI_API_KEY", self.st.session_state.messages[-1]["content"])
        self.assertEqual(ChatOpenAIStub.calls, [])

    def test_environment_api_key_initializes_model(self):
        os.environ["OPENAI_API_KEY"] = "test-key"
        app = self.import_app()

        model = app.initialize_model()

        self.assertIsInstance(model, ChatOpenAIStub)
        self.assertEqual(ChatOpenAIStub.calls[-1]["api_key"], "test-key")


if __name__ == "__main__":
    unittest.main()
