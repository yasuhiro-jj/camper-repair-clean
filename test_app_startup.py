import ast
import pathlib
import unittest


APP_PATH = pathlib.Path(__file__).with_name("streamlit_app.py")
SOURCE = APP_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def find_function(name):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not defined")


class StartupDependencyTests(unittest.TestCase):
    def test_app_does_not_import_local_config_module(self):
        for node in ast.walk(TREE):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                self.assertNotIn("config", imported)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual("config", node.module)

        self.assertNotIn("config.OPENAI_API_KEY", SOURCE)
        self.assertNotIn("config.py", SOURCE)

    def test_api_key_is_loaded_from_deployment_configuration(self):
        helper = find_function("get_openai_api_key")
        constants = {
            node.value
            for node in ast.walk(helper)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("OPENAI_API_KEY", constants)

    def test_missing_knowledge_files_do_not_force_missing_pdf_load(self):
        initialize_database = find_function("initialize_database")
        constants = {
            node.value
            for node in ast.walk(initialize_database)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertNotIn("キャンピングカー修理マニュアル.pdf", constants)

    def test_missing_model_configuration_stops_before_model_invocation(self):
        generate_ai_response = find_function("generate_ai_response")

        has_none_guard = any(
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "model"
            and any(isinstance(op, ast.Is) for op in node.ops)
            and any(isinstance(comparator, ast.Constant) and comparator.value is None for comparator in node.comparators)
            for node in ast.walk(generate_ai_response)
        )

        self.assertTrue(has_none_guard)


if __name__ == "__main__":
    unittest.main()
