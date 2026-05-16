import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("streamlit_app.py").read_text(encoding="utf-8")


class StreamlitAppSourceTest(unittest.TestCase):
    def test_app_does_not_require_untracked_config_module(self):
        self.assertNotIn("import config", SOURCE)
        self.assertNotIn("config.OPENAI_API_KEY", SOURCE)
        self.assertIn("OPENAI_API_KEY", SOURCE)
        self.assertIn("st.secrets", SOURCE)

    def test_database_initialization_does_not_load_missing_fallback_pdf(self):
        missing_pdf_name = "\u30ad\u30e3\u30f3\u30d4\u30f3\u30b0\u30ab\u30fc\u4fee\u7406\u30de\u30cb\u30e5\u30a2\u30eb.pdf"
        self.assertNotIn(missing_pdf_name, SOURCE)
        self.assertIn("return []", SOURCE)


if __name__ == "__main__":
    unittest.main()
