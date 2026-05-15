from pathlib import Path
import unittest


class StreamlitAppSourceTest(unittest.TestCase):
    def test_database_initialization_does_not_require_hard_coded_pdf(self):
        source = Path("streamlit_app.py").read_text(encoding="utf-8")

        self.assertNotIn("キャンピングカー修理マニュアル.pdf", source)


if __name__ == "__main__":
    unittest.main()
