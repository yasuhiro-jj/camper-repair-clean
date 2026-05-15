import sys
import types
import unittest

from app_config import get_openai_api_key


class GetOpenAIApiKeyTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("config", None)

    def test_reads_environment_variable_without_config_module(self):
        sys.modules.pop("config", None)

        api_key = get_openai_api_key(environ={"OPENAI_API_KEY": "env-key"}, secrets={})

        self.assertEqual(api_key, "env-key")

    def test_uses_streamlit_secrets_when_environment_is_missing(self):
        sys.modules.pop("config", None)

        api_key = get_openai_api_key(environ={}, secrets={"OPENAI_API_KEY": "secret-key"})

        self.assertEqual(api_key, "secret-key")

    def test_falls_back_to_optional_config_module(self):
        sys.modules["config"] = types.SimpleNamespace(OPENAI_API_KEY="config-key")

        api_key = get_openai_api_key(environ={}, secrets={})

        self.assertEqual(api_key, "config-key")

    def test_missing_configuration_returns_none_instead_of_crashing(self):
        sys.modules.pop("config", None)

        api_key = get_openai_api_key(environ={}, secrets={})

        self.assertIsNone(api_key)

    def test_blank_values_are_ignored(self):
        sys.modules["config"] = types.SimpleNamespace(OPENAI_API_KEY="  config-key  ")

        api_key = get_openai_api_key(
            environ={"OPENAI_API_KEY": "  "},
            secrets={"OPENAI_API_KEY": ""},
        )

        self.assertEqual(api_key, "config-key")


if __name__ == "__main__":
    unittest.main()
