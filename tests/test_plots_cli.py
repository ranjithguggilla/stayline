import json

import numpy as np
import pandas as pd

from stayline import plots, verify


def test_charts_write_files(tmp_path):
    summary = pd.DataFrame({"model": ["a", "b"], "log_loss": [0.3, 0.4], "log_loss_se": [0.01, 0.02]})
    plots.cv_comparison(summary, "a", tmp_path / "a.png")
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(int)
    plots.calibration(y, p, tmp_path / "b.png")
    imp = pd.DataFrame(
        {"feature": ["x", "noise_0"], "importance": [0.05, 0.0], "sd": [0.01, 0.0], "is_noise": [False, True]}
    )
    plots.importance(imp, tmp_path / "c.png")
    c = pd.DataFrame({"fraction": [0.1, 0.2], "net_per_1000": [1.0, 2.0]})
    plots.budget({"risk only": c}, 0.1, tmp_path / "d.png")
    table = pd.DataFrame(
        {"model": ["cox"], "features": ["without tenure counters"], "c_index": [0.8], "auc_vs_exit_flag": [0.8]}
    )
    plots.survival(table, {"x": (np.array([0.0, 1.0]), np.array([1.0, 0.5]))}, tmp_path / "e.png")
    assert all((tmp_path / f).stat().st_size > 1000 for f in ["a.png", "b.png", "c.png", "d.png", "e.png"])


def test_verify_main_exit_codes(tmp_path):
    m = {
        "test": {"auc": 0.85, "log_loss": 0.31, "ece": 0.05, "lift_top20": 3.2},
        "economics": {"break_even_save_rate": 0.07},
        "reasons": {"lime_overlap_jaccard": 0.46},
        "selection": {"shipped": "logistic"},
    }
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(m))
    b.write_text(json.dumps(m))
    assert verify.main([str(a), str(b)]) == 0
    m["test"]["auc"] = 0.5
    b.write_text(json.dumps(m))
    assert verify.main([str(a), str(b)]) == 1
    assert verify.main([str(a)]) == 2
