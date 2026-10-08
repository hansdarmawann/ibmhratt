"""PSI drift checks: stable on training rows, sensitive to real shifts and unseen values."""

import json

import numpy as np
import pandas as pd
import pytest

from attrition.config import FEATURE_COLUMNS
from attrition.models.artifacts import seal_bundle
from attrition.monitoring.drift import (
    PSI_MODERATE,
    PSI_SHIFT,
    drift_report,
    main,
    psi,
    reference_profile,
)


@pytest.fixture(scope="module")
def train_rows(partitions):
    return partitions[0][FEATURE_COLUMNS]


@pytest.fixture(scope="module")
def profile(train_rows):
    return reference_profile(train_rows)


def by_feature(report: pd.DataFrame) -> pd.DataFrame:
    return report.set_index("feature")


def test_psi_is_zero_for_identical_and_grows_with_difference():
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0
    assert 0 < psi([0.5, 0.5], [0.6, 0.4]) < psi([0.5, 0.5], [0.9, 0.1])
    assert np.isfinite(psi([1.0, 0.0], [0.0, 1.0]))


def test_profile_is_json_serializable_and_sums_to_one(profile, train_rows):
    json.dumps(profile, allow_nan=False)
    assert profile["rows"] == len(train_rows)
    for section in ("numeric", "categorical"):
        for reference in profile[section].values():
            shares = reference["shares"].values() if section == "categorical" else reference["shares"]
            assert sum(shares) == pytest.approx(1)


def test_training_rows_are_stable(profile, train_rows):
    report = drift_report(train_rows, profile)
    assert len(report) == len(FEATURE_COLUMNS)
    assert report["psi"].max() == pytest.approx(0, abs=1e-9)
    assert set(report["status"]) == {"stable"}


def test_shifted_features_are_flagged(profile, partitions):
    holdout = partitions[1][FEATURE_COLUMNS].copy()
    holdout["MonthlyIncome"] *= 3
    holdout["OverTime"] = "Yes"
    holdout["JobRole"] = "Novel role"
    report = by_feature(drift_report(holdout, profile))
    assert report.loc[["MonthlyIncome", "OverTime", "JobRole"], "status"].eq("shift").all()
    assert report.loc["JobRole", "unseen_share"] == 1
    assert report.loc["Age", "psi"] < PSI_SHIFT
    assert report.iloc[0]["psi"] >= report.iloc[-1]["psi"]


def test_holdout_sample_shows_no_shift(profile, partitions):
    report = drift_report(partitions[1][FEATURE_COLUMNS], profile)
    assert (report["psi"] < PSI_SHIFT).all()
    assert (report["psi"] < PSI_MODERATE).mean() > 0.8


def test_missing_values_are_reported_not_binned(profile, partitions):
    holdout = partitions[1][FEATURE_COLUMNS].copy()
    holdout["Age"] = None
    report = by_feature(drift_report(holdout, profile))
    assert report.loc["Age", "status"] == "no data"
    assert report.loc["Age", "current_missing_rate"] == 1
    assert report.iloc[-1]["status"] == "no data"


def test_cli_reads_csv_and_json_and_can_fail_on_shift(bundle_factory, partitions, tmp_path, capsys):
    bundle = bundle_factory()
    holdout = partitions[1][FEATURE_COLUMNS]
    stable_csv = tmp_path / "stable.csv"
    holdout.to_csv(stable_csv, index=False)
    output = tmp_path / "report.csv"
    assert main([str(stable_csv), "--bundle", str(bundle.path), "--output", str(output), "--fail-on-shift"]) == 0
    assert len(pd.read_csv(output)) == len(FEATURE_COLUMNS)
    assert "MonthlyIncome" in capsys.readouterr().out
    shifted = json.loads(holdout.assign(OverTime="Yes").to_json(orient="records"))
    batch_json = tmp_path / "batch.json"
    batch_json.write_text(json.dumps({"employees": shifted}), encoding="utf-8")
    assert main([str(batch_json), "--bundle", str(bundle.path)]) == 0
    assert main([str(batch_json), "--bundle", str(bundle.path), "--fail-on-shift"]) == 1


def test_cli_explains_bundles_without_profile(bundle_factory, partitions, tmp_path):
    bundle = bundle_factory()
    records = tmp_path / "one.json"
    records.write_text(partitions[1][FEATURE_COLUMNS].iloc[:1].to_json(orient="records"), encoding="utf-8")
    profile_path = bundle.path / "reference_profile.json"
    # Simulate an older bundle by resealing it without the profile.
    profile_path.unlink()
    seal_bundle(bundle.path)
    with pytest.raises(SystemExit):
        main([str(records), "--bundle", str(bundle.path)])
