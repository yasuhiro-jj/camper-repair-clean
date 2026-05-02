import unittest

from app_config import get_openai_api_key


class GetOpenAIApiKeyTest(unittest.TestCase):
    def test_reads_openai_api_key_from_environment(self):
        self.assertEqual(
            get_openai_api_key(secrets={}, environ={"OPENAI_API_KEY": "sk-env"}),
            "sk-env",
        )

    def test_environment_takes_precedence_over_secrets(self):
        self.assertEqual(
            get_openai_api_key(
                secrets={"OPENAI_API_KEY": "sk-secret"},
                environ={"OPENAI_API_KEY": "sk-env"},
            ),
            "sk-env",
        )

    def test_reads_nested_streamlit_secret(self):
        self.assertEqual(
            get_openai_api_key(
                secrets={"openai": {"api_key": "sk-nested"}},
                environ={},
            ),
            "sk-nested",
        )

    def test_ignores_empty_values(self):
        self.assertIsNone(
            get_openai_api_key(
                secrets={"OPENAI_API_KEY": "   "},
                environ={"OPENAI_API_KEY": ""},
            )
        )


if __name__ == "__main__":
    unittest.main()
