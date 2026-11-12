"""End-to-end run: compare, select, test, audit, price, save."""

from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import economics as eco
from . import evaluate as ev
from . import plots
from .config import (
    ASSET_DIR,
    BOOTSTRAPS,
    COSTS,
    DEMOGRAPHIC_COLUMNS,
    OUT_DIR,
    RATE_COLUMNS,
    SEED,
    TABLE_DIR,
    TENURE_COLUMNS,
)
from .data import feature_columns, load_raw, split, xy
from .metrics import bootstrap_ci, lift_at, score_all
from .models import build
from .score import Bundle, baseline_defaults

CANDIDATES = ["logistic", "forest", "boosting"]
REFERENCES = ["prior", "logistic_balanced"]


def _r(x, n=4):
    return round(float(x), n)


def lime_agreement(bundle_pipe, train, test, columns, numeric, categorical, n=40, k=3):
    """How often the top risk-raising columns from the swap method and from LIME overlap."""
    from lime.lime_tabular import LimeTabularExplainer

    from .score import Bundle, explain

    cats = {c: sorted(train[c].unique()) for c in categorical}
    enc = train[columns].copy()
    for c in categorical:
        enc[c] = enc[c].map({v: i for i, v in enumerate(cats[c])})

    def decode(arr):
        df = pd.DataFrame(arr, columns=columns)
        for c in categorical:
            df[c] = [cats[c][int(round(v))] for v in df[c]]
        for c in numeric:
            df[c] = df[c].astype(float)
        return df

    explainer = LimeTabularExplainer(
        enc.to_numpy(float),
        feature_names=columns,
        categorical_features=[columns.index(c) for c in categorical],
        class_names=["stay", "leave"],
        discretize_continuous=True,
        random_state=SEED,
    )
    defaults = baseline_defaults(train, columns)
    b = Bundle(bundle_pipe, columns, defaults, 0.0, {})
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(test), size=n, replace=False)
    overlaps = []
    for i in idx:
        row = test.iloc[[i]][columns]
        mine = {r.split(" = ")[0] for r in explain(b, row, k=k)["raises_risk"]}
        x = row.copy()
        for c in categorical:
            x[c] = x[c].map({v: j for j, v in enumerate(cats[c])})
        exp = explainer.explain_instance(
            x.to_numpy(float)[0],
            lambda a: bundle_pipe.predict_proba(decode(a)),
            num_features=len(columns),
            num_samples=800,
        )
        pos = []
        for feat, w in exp.as_list():
            if w > 0:
                col = next(c for c in columns if c in feat)
                pos.append(col)
        from .score import PHRASES

        lime_top = {PHRASES.get(c, c) for c in pos[:k]}
        if mine or lime_top:
            overlaps.append(len(mine & lime_top) / max(1, len(mine | lime_top)))
    return float(np.mean(overlaps)), len(overlaps)


def run(verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    TABLE_DIR.mkdir(exist_ok=True, parents=True)
    ASSET_DIR.mkdir(exist_ok=True, parents=True)

    df = load_raw()
    numeric, categorical = feature_columns(df)
    train, test = split(df)
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)
    columns = list(Xtr.columns)
    M: dict = {
        "data": {
            "employees": len(df),
            "leavers": int(df.Attrition.sum()),
            "attrition_rate": _r(df.Attrition.mean()),
            "train": len(train),
            "test": len(test),
            "test_leavers": int(yte.sum()),
            "features": len(columns),
        }
    }
    log("data", M["data"])

    # 1. model comparison on the training fold only
    summary, folds = ev.compare(Xtr, ytr, numeric, categorical, CANDIDATES + REFERENCES)
    shipped = ev.one_se_pick(summary, CANDIDATES)
    best_mean = summary[summary.model.isin(CANDIDATES)].iloc[0].model
    n_test_cv = len(ytr) // 5
    gap = (
        ev.paired_gap(folds, best_mean, shipped, "log_loss", n_test_cv, len(ytr) - n_test_cv)
        if best_mean != shipped
        else None
    )
    summary.to_csv(TABLE_DIR / "cv_comparison.csv", index=False)
    M["selection"] = {
        "shipped": shipped,
        "best_by_mean": best_mean,
        "cv": {
            r.model: {"log_loss": _r(r.log_loss), "log_loss_se": _r(r.log_loss_se), "auc": _r(r.auc), "ece": _r(r.ece)}
            for r in summary.itertuples()
        },
        "best_minus_shipped_log_loss": None if gap is None else {k: _r(v) for k, v in gap.items()},
    }
    log("shipped", shipped, "best by mean", best_mean)

    # 2. fit shipped on train, score the test fold once
    pipe = build(shipped, numeric, categorical).fit(Xtr, ytr)
    p_te = pipe.predict_proba(Xte)[:, 1]
    sc = score_all(yte, p_te)
    auc_ci = bootstrap_ci(
        yte, p_te, lambda a, b: __import__("sklearn.metrics", fromlist=["x"]).roc_auc_score(a, b), BOOTSTRAPS
    )
    ll_ci = bootstrap_ci(yte, p_te, lambda a, b: score_all(a, b)["log_loss"], 500)
    refs = {}
    for n in CANDIDATES + REFERENCES:
        pp = pipe if n == shipped else build(n, numeric, categorical).fit(Xtr, ytr)
        refs[n] = score_all(yte, pp.predict_proba(Xte)[:, 1])
    prec, rec, lift = lift_at(yte, p_te, 0.20)
    inc_cut = float(train.MonthlyIncome.median())
    rule_te = eco.rule_of_thumb(test, inc_cut)
    from sklearn.metrics import roc_auc_score

    M["test"] = {
        **{k: _r(v) for k, v in sc.items()},
        "auc_ci": [_r(auc_ci[0]), _r(auc_ci[1])],
        "log_loss_ci": [_r(ll_ci[0]), _r(ll_ci[1])],
        "precision_top20": _r(prec),
        "recall_top20": _r(rec),
        "lift_top20": _r(lift),
        "rule_of_thumb_auc": _r(roc_auc_score(yte, rule_te)),
        "all_models": {n: {k: _r(v) for k, v in s.items()} for n, s in refs.items()},
    }
    pd.DataFrame(refs).T.round(4).to_csv(TABLE_DIR / "test_scores.csv")
    log("test", M["test"])

    # 3. ablations and noise-referenced importance
    groups = {
        "tenure counters": TENURE_COLUMNS,
        "pay-rate columns": RATE_COLUMNS,
        "age, gender, marital status": DEMOGRAPHIC_COLUMNS,
    }
    abl = ev.ablation(shipped, Xtr, ytr, numeric, categorical, groups)
    abl.to_csv(TABLE_DIR / "ablation.csv", index=False)
    M["ablation"] = {
        f"{r.dropped} | {r.metric}": {"delta": _r(r.delta), "lo": _r(r.lo), "hi": _r(r.hi)} for r in abl.itertuples()
    }
    imp = ev.noise_importance(pipe, Xtr, ytr, Xte, yte, numeric, categorical)
    imp.to_csv(TABLE_DIR / "importance.csv", index=False)
    noise_top = imp[imp.is_noise].importance.max()
    real = imp[~imp.is_noise]
    M["importance"] = {
        "top": [{"feature": r.feature, "importance": _r(r.importance, 5)} for r in real.head(8).itertuples()],
        "noise_max": _r(noise_top, 5),
        "features_above_noise": int((real.importance - 2 * real.sd > noise_top).sum()),
        "features_total": len(real),
    }

    # 4. survival cross-check
    surv, _risks = ev.survival_check(train, test, numeric, categorical, TENURE_COLUMNS)
    surv.to_csv(TABLE_DIR / "survival.csv", index=False)
    M["survival"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in surv.to_dict("records")]
    from sksurv.nonparametric import kaplan_meier_estimator

    from .data import survival_arrays

    ev_all, t_all = survival_arrays(df)
    km = {}
    for lab, mask in [
        ("works overtime", (df.OverTime == "Yes").to_numpy()),
        ("no overtime", (df.OverTime == "No").to_numpy()),
    ]:
        tt, ss = kaplan_meier_estimator(ev_all[mask], t_all[mask])
        km[lab] = (tt, ss)
    M["survival_km"] = {lab: _r(float(np.interp(5.5, tt, ss))) for lab, (tt, ss) in km.items()}

    # 5. money: choose the call budget on out-of-fold training scores, price it on the test fold
    oof = cross_val_predict(
        build(shipped, numeric, categorical),
        Xtr,
        ytr,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        method="predict_proba",
    )[:, 1]
    repl_tr = eco.replacement_cost(train.MonthlyIncome.to_numpy())
    grid = np.round(np.arange(0.02, 1.0001, 0.02), 2)
    oof_curve = eco.budget_curve(oof * repl_tr, repl_tr, ytr.astype(float), grid)
    chosen = float(oof_curve.loc[oof_curve.net_per_1000.idxmax(), "fraction"])
    repl_te = eco.replacement_cost(test.MonthlyIncome.to_numpy())
    leaves = yte.astype(float)
    pol = eco.policy_table(p_te, repl_te, rule_te, leaves, chosen)
    pol_default = eco.policy_table(p_te, repl_te, rule_te, leaves, COSTS.budget_fraction)
    pol.to_csv(TABLE_DIR / "policies.csv", index=False)
    pol_default.to_csv(TABLE_DIR / "policies_at_20pct.csv", index=False)
    scores = eco.strategies(p_te, repl_te, rule_te, leaves)
    curves = {n: eco.budget_curve(s, repl_te, leaves, grid) for n, s in scores.items() if n != "oracle (hindsight)"}
    curves["random calls"] = pd.DataFrame(
        {
            "fraction": grid,
            "net_per_1000": [
                (f * (leaves * repl_te).mean() * COSTS.save_rate - f * COSTS.cost_per_conversation) * 1000 for f in grid
            ],
        }
    )
    sel = eco.top_fraction(p_te * repl_te, chosen)
    be = eco.break_even_save_rate(p_te * repl_te, repl_te, leaves, chosen)
    save_sweep = {
        f"{s:.2f}": _r(eco.net_value(leaves, repl_te, sel, save_rate=s) / len(test) * 1000, 0)
        for s in (0.05, 0.10, 0.15, 0.25, 0.40)
    }
    M["economics"] = {
        "assumptions": {
            "replacement_share_of_annual_pay": COSTS.replacement_share_of_annual_pay,
            "cost_per_conversation": COSTS.cost_per_conversation,
            "save_rate": COSTS.save_rate,
        },
        "chosen_budget_fraction": chosen,
        "mean_replacement_cost": _r(repl_te.mean(), 0),
        "policies_at_chosen": {
            r.policy: {"net_per_1000": _r(r.net_per_1000, 0), "lo": _r(r.lo, 0), "hi": _r(r.hi, 0)}
            for r in pol.itertuples()
        },
        "policies_at_20pct": {
            r.policy: {"net_per_1000": _r(r.net_per_1000, 0), "lo": _r(r.lo, 0), "hi": _r(r.hi, 0)}
            for r in pol_default.itertuples()
        },
        "break_even_save_rate": _r(be),
        "net_per_1000_by_save_rate": save_sweep,
        "leavers_reached_at_chosen": _r(float(leaves[sel].sum() / leaves.sum())),
        "people_called": int(sel.sum()),
    }
    log("economics", M["economics"]["policies_at_chosen"])

    # 6. who the list reaches
    aud = []
    bands = pd.cut(test.Age, [0, 29, 44, 100], labels=["under 30", "30 to 44", "45 and over"])
    for gname, series in [
        ("Gender", test.Gender),
        ("MaritalStatus", test.MaritalStatus),
        ("age band", bands.astype(str)),
    ]:
        for g in sorted(series.unique()):
            m = (series == g).to_numpy()
            aud.append(
                {
                    "group": f"{gname}: {g}",
                    "employees": int(m.sum()),
                    "leavers": int(leaves[m].sum()),
                    "share_called": _r(sel[m].mean()),
                    "share_of_leavers_reached": _r(sel[m & (leaves == 1)].sum() / max(1, leaves[m].sum())),
                }
            )
    pd.DataFrame(aud).to_csv(TABLE_DIR / "group_audit.csv", index=False)
    M["group_audit"] = aud

    # 7. reasons: swap method versus LIME
    overlap, n_lime = lime_agreement(pipe, train, test, columns, numeric, categorical)
    M["reasons"] = {"lime_overlap_jaccard": _r(overlap), "employees_compared": n_lime}

    # 8. charts, bundle
    plots.cv_comparison(summary, shipped, ASSET_DIR / "cv_comparison.png")
    plots.calibration(yte, p_te, ASSET_DIR / "calibration.png")
    plots.importance(imp, ASSET_DIR / "importance.png")
    plots.budget(curves, chosen, ASSET_DIR / "budget.png")
    plots.survival(surv, km, ASSET_DIR / "survival.png")

    final = build(shipped, numeric, categorical).fit(pd.concat([Xtr, Xte]), np.concatenate([ytr, yte]))
    thresholds = (float(np.quantile(oof, 0.60)), float(np.quantile(oof, 0.90)))
    bundle = Bundle(
        final,
        columns,
        baseline_defaults(train, columns),
        inc_cut,
        {"band_thresholds": list(thresholds), "job_roles": sorted(train.JobRole.unique()), "model": shipped},
    )
    joblib.dump(bundle, OUT_DIR / "bundle.joblib")
    M["bundle"] = {"band_thresholds": [_r(t) for t in thresholds], "refit_on": "train and test together"}
    M["runtime_seconds"] = round(time.time() - t0)
    (OUT_DIR / "metrics.json").write_text(json.dumps(M, indent=2, default=str))
    log("done in", M["runtime_seconds"], "s")
    return M


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
