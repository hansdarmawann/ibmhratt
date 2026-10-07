"""Execute Streamlit and its prediction form without a browser or server."""

from streamlit.testing.v1 import AppTest

from src.config import ROOT


def test_dashboard_renders_and_predicts(monkeypatch, artifact):
    monkeypatch.setattr("src.config.MODEL_PATH", artifact)
    dashboard = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    assert not dashboard.exception
    dashboard.button[0].click().run()
    assert not dashboard.exception
    labels = [metric.label for metric in dashboard.metric]
    assert "Attrition probability" in labels
    assert "Selected threshold" in labels
