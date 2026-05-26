import ast
import pathlib
import unittest


APP_PATH = pathlib.Path(__file__).with_name("streamlit_app.py")


class StartupConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_app_does_not_import_untracked_config_module(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported_names = {alias.name for alias in node.names}
                self.assertNotIn("config", imported_names)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual("config", node.module)

        self.assertNotIn("config.OPENAI_API_KEY", self.source)

    def test_api_key_comes_from_environment_or_streamlit_secrets(self):
        self.assertIn("def get_openai_api_key", self.source)
        self.assertIn("OPENAI_API_KEY", self.source)
        self.assertIn("os.environ", self.source)
        self.assertIn("st.secrets", self.source)

    def test_missing_knowledge_files_do_not_load_untracked_fallback_pdf(self):
        self.assertNotIn("キャンピングカー修理マニュアル.pdf", self.source)
        self.assertRegex(
            self.source,
            r"if not documents:\s*\n\s*return \[\]",
        )

    def test_generation_stops_when_model_is_not_configured(self):
        pattern = (
            r"(?s)"
            r"model = build_workflow\(\)\s*\n"
            r"\s*if model is None:\s*\n"
            r"\s*return\s*\n"
            r".*?model\.invoke\(messages\)"
        )
        self.assertRegex(self.source, pattern)


if __name__ == "__main__":
    unittest.main()
