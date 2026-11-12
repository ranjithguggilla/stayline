import numpy as np

from stayline.metrics import bootstrap_ci, corrected_se, ece, lift_at, score_all


def test_ece_perfect_and_bad():
    y = np.array([0, 0, 1, 1] * 25)
    assert ece(y, np.where(y == 1, 0.999, 0.001)) < 0.01
    assert ece(y, np.full(len(y), 0.9)) > 0.3


def test_score_all_keys():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = score_all(y, np.array([0.2, 0.8, 0.3, 0.6, 0.7, 0.1]))
    assert set(s) == {"log_loss", "auc", "avg_precision", "brier", "ece"}
    assert s["auc"] == 1.0


def test_corrected_se_larger_than_naive():
    x = np.random.default_rng(1).normal(size=25)
    assert corrected_se(x, 20, 80) > x.std(ddof=1) / 5


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    from sklearn.metrics import roc_auc_score

    lo, hi = bootstrap_ci(y, p, roc_auc_score, n=200)
    assert lo < roc_auc_score(y, p) < hi


def test_lift_at_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    prec, rec, lift = lift_at(y, p, 0.2)
    assert (prec, rec) == (1.0, 1.0) and lift == 5.0
