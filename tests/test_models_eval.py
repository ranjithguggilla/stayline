import numpy as np
import pandas as pd
import pytest

from stayline import evaluate as ev
from stayline.models import build


def _xy(toy):
    X = toy.drop(columns="Attrition")
    return X, toy.Attrition.to_numpy()


@pytest.mark.parametrize("name", ["prior", "logistic", "logistic_balanced", "forest", "boosting"])
def test_models_fit_and_predict(toy, name):
    X, y = _xy(toy)
    p = build(name, ["x"], ["g"]).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build("nope", [], [])


def test_pipeline_handles_unseen_category(toy):
    X, y = _xy(toy)
    pipe = build("logistic", ["x"], ["g"]).fit(X, y)
    new = X.head(3).copy()
    new["g"] = "zzz"
    assert np.isfinite(pipe.predict_proba(new)).all()


def test_one_se_prefers_simpler():
    s = pd.DataFrame(
        {
            "model": ["boosting", "logistic", "forest"],
            "log_loss": [0.34, 0.345, 0.36],
            "log_loss_se": [0.02, 0.02, 0.02],
        }
    )
    assert ev.one_se_pick(s, ["logistic", "forest", "boosting"]) == "logistic"


def test_one_se_keeps_clear_winner():
    s = pd.DataFrame({"model": ["boosting", "logistic"], "log_loss": [0.30, 0.40], "log_loss_se": [0.01, 0.01]})
    assert ev.one_se_pick(s, ["logistic", "boosting"]) == "boosting"


def test_compare_returns_all_models(toy):
    X, y = _xy(toy)
    summary, folds = ev.compare(X, y, ["x"], ["g"], ["prior", "logistic"])
    assert set(summary.model) == {"prior", "logistic"} and len(folds) == 50
    assert summary.set_index("model").loc["logistic", "log_loss"] < summary.set_index("model").loc["prior", "log_loss"]


def test_paired_gap_zero_for_same_model(toy):
    X, y = _xy(toy)
    _, folds = ev.compare(X, y, ["x"], ["g"], ["logistic"])
    g = ev.paired_gap(folds, "logistic", "logistic", "log_loss", 60, 240)
    assert g["gap"] == 0


def test_noise_columns_rank_low(toy):
    X, y = _xy(toy)
    pipe = build("logistic", ["x"], ["g"]).fit(X, y)
    imp = ev.noise_importance(pipe, X, y, X, y, ["x"], ["g"], n_noise=2, repeats=5)
    assert imp.iloc[0].feature == "x" and imp[imp.is_noise].importance.max() < imp.iloc[0].importance
