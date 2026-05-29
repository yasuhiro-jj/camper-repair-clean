import ast
import pathlib
import unittest


APP_SOURCE = pathlib.Path(__file__).with_name("streamlit_app.py").read_text()
APP_TREE = ast.parse(APP_SOURCE)


class StartupRegressionTests(unittest.TestCase):
    def test_config_module_is_not_required_at_import_time(self):
        imported_modules = {
            alias.name
            for node in APP_TREE.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        self.assertNotIn("config", imported_modules)
        self.assertIn('os.environ.get("OPENAI_API_KEY")', APP_SOURCE)
        self.assertIn('st.secrets.get("OPENAI_API_KEY")', APP_SOURCE)

    def test_missing_knowledge_files_do_not_load_untracked_pdf(self):
        self.assertIn("if not documents:\n        return documents", APP_SOURCE)
        missing_pdf = "\u30ad\u30e3\u30f3\u30d4\u30f3\u30b0\u30ab\u30fc\u4fee\u7406\u30de\u30cb\u30e5\u30a2\u30eb.pdf"
        self.assertNotIn(f'os.path.join(main_path, "{missing_pdf}")', APP_SOURCE)

    def test_missing_model_configuration_does_not_invoke_none(self):
        self.assertIn("model = build_workflow()\n        if model is None:\n            return", APP_SOURCE)
        self.assertIn("response = model.invoke(messages)", APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
