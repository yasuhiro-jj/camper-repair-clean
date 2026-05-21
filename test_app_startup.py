import ast
import unittest
from pathlib import Path


APP_PATH = Path(__file__).with_name("streamlit_app.py")


class StartupRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_app_does_not_require_local_config_module_at_import_time(self):
        imports = [
            alias.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        self.assertNotIn("config", imports)

    def test_openai_key_comes_from_deployment_configuration(self):
        self.assertIn("def get_openai_api_key", self.source)
        self.assertIn('os.environ.get("OPENAI_API_KEY")', self.source)
        self.assertIn("st.secrets", self.source)
        self.assertNotIn("config.OPENAI_API_KEY", self.source)

    def test_missing_knowledge_files_do_not_trigger_hard_coded_pdf_load(self):
        self.assertNotIn("キャンピングカー修理マニュアル.pdf", self.source)
        self.assertIn("if not documents:\n        return []", self.source)


if __name__ == "__main__":
    unittest.main()
