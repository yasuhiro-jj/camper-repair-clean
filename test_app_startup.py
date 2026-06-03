import importlib.util
import os
import sys
import types
import unittest


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
        self.errors = []
        self.infos = []
        self.markdowns = []

    def set_page_config(self, **kwargs):
        self.page_config = kwargs

    def markdown(self, *args, **kwargs):
        self.markdowns.append((args, kwargs))

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.infos.append(message)

    def cache_resource(self, func=None, **kwargs):
        if func is not None:
            return func
        return lambda wrapped: wrapped


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def install_fake_dependencies(streamlit_module):
    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = FakeChatOpenAI
    langchain_openai.OpenAIEmbeddings = object

    messages = types.ModuleType("langchain_core.messages")
    messages.BaseMessage = FakeMessage
    messages.HumanMessage = FakeMessage
    messages.AIMessage = FakeMessage

    loaders = types.ModuleType("langchain_community.document_loaders")
    loaders.PyPDFLoader = object
    loaders.TextLoader = object

    sys.modules["streamlit"] = streamlit_module
    sys.modules["langchain_openai"] = langchain_openai
    sys.modules["langchain_core"] = types.ModuleType("langchain_core")
    sys.modules["langchain_core.messages"] = messages
    sys.modules["langchain_community"] = types.ModuleType("langchain_community")
    sys.modules["langchain_community.document_loaders"] = loaders


def load_app_module():
    streamlit_module = FakeStreamlit()
    install_fake_dependencies(streamlit_module)
    sys.modules.pop("config", None)

    spec = importlib.util.spec_from_file_location("streamlit_app_under_test", "streamlit_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, streamlit_module


class StartupRegressionTests(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.get("OPENAI_API_KEY")
        os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self):
        if self.original_env is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self.original_env

        for name in [
            "streamlit_app_under_test",
            "streamlit",
            "langchain_openai",
            "langchain_core",
            "langchain_core.messages",
            "langchain_community",
            "langchain_community.document_loaders",
            "config",
        ]:
            sys.modules.pop(name, None)

    def test_module_imports_without_local_config_file(self):
        module, _ = load_app_module()

        self.assertTrue(hasattr(module, "main"))

    def test_api_key_uses_environment_without_config_file(self):
        os.environ["OPENAI_API_KEY"] = "env-key"
        module, _ = load_app_module()

        self.assertEqual(module.get_openai_api_key(), "env-key")
        self.assertEqual(module.initialize_model().kwargs["api_key"], "env-key")

    def test_missing_api_key_does_not_invoke_none_model(self):
        module, streamlit_module = load_app_module()
        streamlit_module.session_state.messages = [{"role": "user", "content": "バッテリーが上がった"}]
        module.initialize_database = lambda: []
        module.build_workflow = lambda: None

        module.generate_ai_response("バッテリーが上がった")

        self.assertIn("OpenAI APIキーが未設定", streamlit_module.errors[-1])
        self.assertEqual(streamlit_module.session_state.messages[-1]["role"], "assistant")
        self.assertIn("OPENAI_API_KEY", streamlit_module.session_state.messages[-1]["content"])

    def test_database_initialization_allows_no_bundled_documents(self):
        module, _ = load_app_module()
        module.glob.glob = lambda pattern: []

        self.assertEqual(module.initialize_database(), [])


if __name__ == "__main__":
    unittest.main()
