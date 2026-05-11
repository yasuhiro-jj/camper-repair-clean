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


def cache_resource(func=None, **_kwargs):
    if func is None:
        return lambda wrapped: wrapped
    return func


def install_dependency_stubs(secrets=None):
    st = types.SimpleNamespace(
        cache_resource=cache_resource,
        set_page_config=lambda **_kwargs: None,
        markdown=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        session_state=SessionState(),
        secrets=secrets or {},
    )

    langchain_openai = types.SimpleNamespace(
        ChatOpenAI=lambda **kwargs: types.SimpleNamespace(kwargs=kwargs),
        OpenAIEmbeddings=object,
    )
    langchain_core_messages = types.SimpleNamespace(
        BaseMessage=object,
        HumanMessage=lambda content: types.SimpleNamespace(content=content),
        AIMessage=lambda content: types.SimpleNamespace(content=content),
    )
    document_loaders = types.SimpleNamespace(
        PyPDFLoader=object,
        TextLoader=object,
    )

    return {
        "streamlit": st,
        "langchain_openai": langchain_openai,
        "langchain_core": types.ModuleType("langchain_core"),
        "langchain_core.messages": langchain_core_messages,
        "langchain_community": types.ModuleType("langchain_community"),
        "langchain_community.document_loaders": document_loaders,
    }


def import_app(secrets=None):
    sys.modules.pop("streamlit_app", None)
    sys.modules.pop("config", None)
    with mock.patch.dict(sys.modules, install_dependency_stubs(secrets), clear=False):
        return importlib.import_module("streamlit_app")


class StreamlitAppConfigurationTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("streamlit_app", None)

    def test_import_does_not_require_untracked_config_module(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            app = import_app()

        self.assertIsNone(app.get_openai_api_key())

    def test_get_openai_api_key_prefers_environment(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            app = import_app({"OPENAI_API_KEY": "secret-key"})
            self.assertEqual(app.get_openai_api_key(), "env-key")

    def test_get_openai_api_key_uses_streamlit_secrets(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            app = import_app({"OPENAI_API_KEY": "secret-key"})

        self.assertEqual(app.get_openai_api_key(), "secret-key")


if __name__ == "__main__":
    unittest.main()
