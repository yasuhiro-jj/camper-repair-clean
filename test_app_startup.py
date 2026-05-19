import ast
import pathlib
import unittest


APP_PATH = pathlib.Path(__file__).with_name("streamlit_app.py")


class StartupConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_app_does_not_require_local_config_module_at_import_time(self):
        imported_modules = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        self.assertNotIn("config", imported_modules)
        self.assertIn("def get_openai_api_key", self.source)
        self.assertIn('os.getenv("OPENAI_API_KEY")', self.source)

    def test_missing_documents_do_not_load_untracked_fallback_pdf(self):
        self.assertNotIn("キャンピングカー修理マニュアル.pdf", self.source)
        self.assertIn("if not documents:\n        return []", self.source)


if __name__ == "__main__":
    unittest.main()
