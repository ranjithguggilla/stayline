import json

from stayline import evaluate as ev
from stayline import verify
from stayline.config import TENURE_COLUMNS


def test_survival_check_runs_on_subsample(parts):
    train, test, numeric, categorical = parts
    tab, risks = ev.survival_check(train.head(500), test.head(200), numeric, categorical, TENURE_COLUMNS)
    assert len(tab) == 4 and tab.c_index.between(0.4, 1.0).all()
    assert len(risks) == 4


def test_tenure_counters_inflate_concordance(parts):
    train, test, numeric, categorical = parts
    tab, _ = ev.survival_check(train, test, numeric, categorical, TENURE_COLUMNS)
    cox = tab[tab.model == "cox"].set_index("features")
    assert cox.loc["with tenure counters", "c_index"] > cox.loc["without tenure counters", "c_index"]


def test_verify_compare_flags_drift():
    a = {
        "test": {"auc": 0.85, "log_loss": 0.31, "ece": 0.05, "lift_top20": 3.2},
        "economics": {"break_even_save_rate": 0.07},
        "reasons": {"lime_overlap_jaccard": 0.46},
        "selection": {"shipped": "logistic"},
    }
    assert verify.compare(a, a) == []
    b = json.loads(json.dumps(a))
    b["test"]["auc"] = 0.80
    assert any("auc" in line for line in verify.compare(a, b))


def test_spearman_perfect():
    assert ev.spearman([1, 2, 3], [10, 20, 30]) == 1.0
