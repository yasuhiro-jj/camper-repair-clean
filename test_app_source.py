from pathlib import Path


SOURCE = Path(__file__).with_name("streamlit_app.py").read_text(encoding="utf-8")


def test_app_does_not_require_untracked_config_module():
    assert "import config" not in SOURCE
    assert "config.OPENAI_API_KEY" not in SOURCE
    assert "OPENAI_API_KEY" in SOURCE
    assert "st.secrets" in SOURCE


def test_database_initialization_does_not_load_missing_fallback_pdf():
    missing_pdf_name = "\u30ad\u30e3\u30f3\u30d4\u30f3\u30b0\u30ab\u30fc\u4fee\u7406\u30de\u30cb\u30e5\u30a2\u30eb.pdf"
    assert missing_pdf_name not in SOURCE
    assert "return []" in SOURCE
