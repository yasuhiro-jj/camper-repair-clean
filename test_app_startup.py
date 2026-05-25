import ast
import pathlib
import unittest


APP_PATH = pathlib.Path(__file__).with_name("streamlit_app.py")
SOURCE = APP_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def function_source(name):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SOURCE, node)
    raise AssertionError(f"{name} not found")


class StartupConfigurationTests(unittest.TestCase):
    def test_app_does_not_import_local_config_module(self):
        imported_modules = []
        for node in ast.walk(TREE):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        self.assertNotIn("config", imported_modules)

    def test_openai_key_uses_deployment_configuration(self):
        model_source = function_source("initialize_model")

        self.assertIn("get_openai_api_key()", model_source)
        self.assertNotIn("config.", model_source)
        self.assertIn("OPENAI_API_KEY", function_source("get_openai_api_key"))

    def test_missing_knowledge_files_do_not_load_untracked_pdf(self):
        database_source = function_source("initialize_database")

        self.assertNotIn("キャンピングカー修理マニュアル.pdf", database_source)
        self.assertIn("return []", database_source)

    def test_missing_model_configuration_stops_before_invoke(self):
        response_source = function_source("generate_ai_response")

        guard_index = response_source.index("if model is None:")
        invoke_index = response_source.index("model.invoke")
        self.assertLess(guard_index, invoke_index)


if __name__ == "__main__":
    unittest.main()
