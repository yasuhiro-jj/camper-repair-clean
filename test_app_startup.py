import ast
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).with_name("streamlit_app.py")
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")


class StartupRegressionTests(unittest.TestCase):
    def test_config_module_is_not_required_at_import_time(self):
        tree = ast.parse(SOURCE)
        top_level_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]

        imported_modules = set()
        for node in top_level_imports:
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            else:
                imported_modules.add(node.module)

        self.assertNotIn("config", imported_modules)
        self.assertNotIn("config.OPENAI_API_KEY", SOURCE)

    def test_api_key_lookup_supports_deployed_configuration_sources(self):
        self.assertIn("def get_openai_api_key", SOURCE)
        self.assertIn('os.getenv("OPENAI_API_KEY")', SOURCE)
        self.assertIn('st.secrets.get("OPENAI_API_KEY")', SOURCE)
        self.assertIn("import config as local_config", SOURCE)
        self.assertIn('if e.name != "config":', SOURCE)

    def test_missing_knowledge_files_do_not_load_untracked_fallback_pdf(self):
        self.assertNotIn("キャンピングカー修理マニュアル.pdf", SOURCE)
        self.assertIn("if not documents:\n        return []", SOURCE)

    def test_missing_model_is_not_invoked(self):
        guard_index = SOURCE.index("if model is None:")
        invoke_index = SOURCE.index("response = model.invoke(messages)")

        self.assertLess(guard_index, invoke_index)
        self.assertIn('st.session_state.messages.append({', SOURCE[guard_index:invoke_index])
        self.assertIn('"role": "assistant"', SOURCE[guard_index:invoke_index])


if __name__ == "__main__":
    unittest.main()
