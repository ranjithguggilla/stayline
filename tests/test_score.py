import joblib
import numpy as np
import pytest

from stayline import score
from stayline.config import OUT_DIR
from stayline.data import xy
from stayline.models import build


@pytest.fixture(scope="module")
def bundle(parts):
    train, _, numeric, categorical = parts
    X, y = xy(train)
    pipe = build("logistic", numeric, categorical).fit(X, y)
    cols = list(X.columns)
    return score.Bundle(
        pipe,
        cols,
        score.baseline_defaults(train, cols),
        float(train.MonthlyIncome.median()),
        {"band_thresholds": [0.1, 0.3], "job_roles": sorted(train.JobRole.unique())},
    )


def test_defaults_cover_columns(bundle):
    assert set(bundle.defaults) == set(bundle.columns)


def test_score_employee_shape(bundle):
    out = score.score_employee(bundle, {"OverTime": "Yes", "MonthlyIncome": 2500, "YearsSinceLastPromotion": 6})
    assert 0 < out["attrition_probability"] < 1
    assert out["risk_band"] in {"low", "medium", "high"}
    assert set(out["reasons"]) == {"raises_risk", "lowers_risk"}


def test_overtime_raises_risk(bundle):
    a = score.score_employee(bundle, {"OverTime": "Yes"})["attrition_probability"]
    b = score.score_employee(bundle, {"OverTime": "No"})["attrition_probability"]
    assert a > b


def test_reasons_name_overtime(bundle):
    out = score.score_employee(bundle, {"OverTime": "Yes", "MonthlyIncome": 2000, "StockOptionLevel": 0})
    assert any("overtime" in r for r in out["reasons"]["raises_risk"])


def test_call_value_sign_follows_risk(bundle):
    low = score.score_employee(bundle, {"OverTime": "No", "MonthlyIncome": 15000, "StockOptionLevel": 3, "Age": 50})
    high = score.score_employee(
        bundle, {"OverTime": "Yes", "MonthlyIncome": 4000, "MaritalStatus": "Single", "StockOptionLevel": 0}
    )
    assert high["expected_net_value_of_a_call_usd"] > low["expected_net_value_of_a_call_usd"]


def test_band_thresholds():
    assert score.band(0.05, (0.1, 0.3)) == "low"
    assert score.band(0.2, (0.1, 0.3)) == "medium"
    assert score.band(0.5, (0.1, 0.3)) == "high"


def test_unknown_field_is_ignored(bundle):
    row = score.complete_record(bundle, {"NotAColumn": 1})
    assert list(row.columns) == bundle.columns and np.isfinite(row.select_dtypes("number").to_numpy()).all()


def test_app_builds_if_bundle_present():
    path = OUT_DIR / "bundle.joblib"
    if not path.exists():
        pytest.skip("bundle not built yet")
    from stayline.app import build_app

    assert build_app(joblib.load(path)) is not None
