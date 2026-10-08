"""Exercise bundle refresh and profile-specific explanations in the real UI."""

from streamlit.testing.v1 import AppTest

from attrition.config import ROOT


def dashboard():
    return AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()


def test_dashboard_renders_predicts_and_explains(monkeypatch, bundle_factory, tmp_path):
    bundle = bundle_factory()
    monkeypatch.setattr("attrition.config.CURRENT_RUN", tmp_path / "models/current.json")
    captured = []

    def explanation(pipeline, profile, background):
        captured.append((pipeline.attrition_metadata_["run_id"], profile.copy(), len(background)))
        return {"status": "generated", "units": "log-odds", "base_value": -1, "other_contribution": 0.1,
                "contributions": [{"feature": "Age", "contribution": 0.2, "direction": "Higher"}]}

    monkeypatch.setattr("attrition.models.local_explain.explain_profile", explanation)
    app = dashboard()
    assert not app.exception
    next(widget for widget in app.number_input if widget.label == "Age").set_value(45)
    app.button[0].click().run()
    assert not app.exception
    assert captured[0][0] == bundle.run_id
    assert captured[0][1]["Age"] == 45
    assert captured[0][2] == 20
    assert "Attrition probability" in [metric.label for metric in app.metric]
    assert any("contribution" in table.value.columns for table in app.dataframe)
    second = bundle_factory(threshold=0.8)
    app.run()
    app.button[0].click().run()
    assert not app.exception
    assert captured[-1][0] == second.run_id
    assert next(m for m in app.metric if m.label == "Selected threshold").value == "0.80"


def test_dashboard_optional_explanation_failure(monkeypatch, bundle_factory, tmp_path):
    bundle_factory()
    monkeypatch.setattr("attrition.config.CURRENT_RUN", tmp_path / "models/current.json")
    monkeypatch.setattr("attrition.models.local_explain.explain_profile", lambda *args: {"status": "unavailable"})
    app = dashboard()
    app.button[0].click().run()
    assert not app.exception
    assert any("Optional SHAP explanation is unavailable" in info.value for info in app.info)
    assert "Attrition probability" in [metric.label for metric in app.metric]


def test_dashboard_without_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr("attrition.config.CURRENT_RUN", tmp_path / "missing.json")
    app = dashboard()
    assert not app.exception
    assert not app.button
    assert any("complete model/report bundle" in info.value for info in app.info)


def test_dashboard_rejects_corrupt_bundle(monkeypatch, bundle_factory, tmp_path):
    bundle = bundle_factory()
    (bundle.path / "employee.json").write_text("{}")
    monkeypatch.setattr("attrition.config.CURRENT_RUN", tmp_path / "models/current.json")
    app = dashboard()
    assert not app.exception
    assert any("could not be verified" in error.value for error in app.error)
    assert not app.button
