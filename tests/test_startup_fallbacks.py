import importlib
import sys
import types


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _install_import_stubs(monkeypatch):
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = _SessionState()
    streamlit.secrets = {}
    streamlit.set_page_config = lambda **kwargs: None
    streamlit.markdown = lambda *args, **kwargs: None
    streamlit.cache_resource = lambda func: func
    streamlit.error = lambda *args, **kwargs: None
    streamlit.info = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "streamlit", streamlit)

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = object
    langchain_openai.OpenAIEmbeddings = object
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai)

    langchain_core = types.ModuleType("langchain_core")
    langchain_core_messages = types.ModuleType("langchain_core.messages")
    langchain_core_messages.BaseMessage = object
    langchain_core_messages.HumanMessage = object
    langchain_core_messages.AIMessage = object
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", langchain_core_messages)

    loaders = types.ModuleType("langchain_community.document_loaders")

    class _FailingLoader:
        def __init__(self, path, *args, **kwargs):
            raise AssertionError(f"unexpected loader call for {path}")

    loaders.PyPDFLoader = _FailingLoader
    loaders.TextLoader = _FailingLoader
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", loaders)


def test_app_imports_without_local_config_file(monkeypatch):
    _install_import_stubs(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sys.modules.pop("config", None)
    sys.modules.pop("streamlit_app", None)

    app = importlib.import_module("streamlit_app")

    assert app.get_openai_api_key() == ""


def test_initialize_database_returns_empty_list_when_no_documents(monkeypatch):
    _install_import_stubs(monkeypatch)
    sys.modules.pop("streamlit_app", None)
    app = importlib.import_module("streamlit_app")

    assert app.initialize_database() == []
