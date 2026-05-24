import ast
import pathlib
import unittest


APP_PATH = pathlib.Path(__file__).with_name("streamlit_app.py")
SOURCE = APP_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


class StartupRegressionTests(unittest.TestCase):
    def test_app_does_not_require_local_config_module(self):
        for node in ast.walk(TREE):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                self.assertNotIn("config", imported)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "config")
            elif isinstance(node, ast.Name):
                self.assertNotEqual(node.id, "config")

    def test_api_key_can_be_loaded_from_deployable_sources(self):
        self.assertIn("def get_openai_api_key", SOURCE)
        self.assertIn('os.environ.get("OPENAI_API_KEY")', SOURCE)
        self.assertIn("st.secrets.get", SOURCE)

    def test_missing_knowledge_files_do_not_load_untracked_fallback_pdf(self):
        self.assertNotIn("if not documents:\n        pdf_path", SOURCE)
        self.assertIn("if not documents:\n        return []", SOURCE)

    def test_missing_model_stops_before_invocation(self):
        guard_index = SOURCE.index("if model is None:")
        invoke_index = SOURCE.index("response = model.invoke(messages)")
        self.assertLess(guard_index, invoke_index)


if __name__ == "__main__":
    unittest.main()
