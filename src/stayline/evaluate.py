"""Model comparison, ablations, noise-referenced importance and the survival cross-check."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, CV_REPEATS, SEED, TARGET
from .metrics import corrected_se, score_all
from .models import COMPLEXITY, build


def split_columns(columns: list[str], numeric: list[str], categorical: list[str]) -> tuple[list[str], list[str]]:
    return [c for c in numeric if c in columns], [c for c in categorical if c in columns]


def cv_scores(name: str, X: pd.DataFrame, y: np.ndarray, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Repeated stratified CV, one row per fold."""
    num, cat = split_columns(list(X.columns), numeric, categorical)
    rkf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=SEED)
    rows = []
    for k, (tr, te) in enumerate(rkf.split(X, y)):
        pipe = build(name, num, cat).fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        rows.append({"model": name, "fold": k, **score_all(y[te], p)})
    return pd.DataFrame(rows)


def compare(X: pd.DataFrame, y: np.ndarray, numeric: list[str], categorical: list[str], names: list[str]):
    folds = pd.concat([cv_scores(n, X, y, numeric, categorical) for n in names], ignore_index=True)
    n_test = len(y) // CV_FOLDS
    n_train = len(y) - n_test
    rows = []
    for n in names:
        f = folds[folds.model == n]
        rows.append(
            {
                "model": n,
                "log_loss": f.log_loss.mean(),
                "log_loss_se": corrected_se(f.log_loss.to_numpy(), n_test, n_train),
                "auc": f.auc.mean(),
                "auc_se": corrected_se(f.auc.to_numpy(), n_test, n_train),
                "avg_precision": f.avg_precision.mean(),
                "brier": f.brier.mean(),
                "ece": f.ece.mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True), folds


def one_se_pick(summary: pd.DataFrame, candidates: list[str]) -> str:
    """Simplest candidate whose CV log loss is within one corrected standard error of the best."""
    s = summary[summary.model.isin(candidates)]
    best = s.loc[s.log_loss.idxmin()]
    ok = s[s.log_loss <= best.log_loss + best.log_loss_se]
    return str(ok.sort_values(["model"], key=lambda c: c.map(COMPLEXITY)).iloc[0].model)


def paired_gap(folds: pd.DataFrame, a: str, b: str, metric: str, n_test: int, n_train: int) -> dict[str, float]:
    """Mean of (a - b) per fold with a corrected standard error and a 95% interval."""
    fa = folds[folds.model == a].sort_values("fold")[metric].to_numpy()
    fb = folds[folds.model == b].sort_values("fold")[metric].to_numpy()
    d = fa - fb
    se = corrected_se(d, n_test, n_train)
    m = float(d.mean())
    return {"gap": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se}


def ablation(name: str, X: pd.DataFrame, y: np.ndarray, numeric, categorical, groups: dict[str, list[str]]):
    """Drop a column group, rerun CV, report the change in log loss and AUC with a corrected interval."""
    base = cv_scores(name, X, y, numeric, categorical)
    n_test = len(y) // CV_FOLDS
    n_train = len(y) - n_test
    rows = []
    for label, cols in groups.items():
        keep = [c for c in X.columns if c not in cols]
        alt = cv_scores(name, X[keep], y, numeric, categorical)
        for metric in ("log_loss", "auc"):
            d = alt[metric].to_numpy() - base[metric].to_numpy()
            se = corrected_se(d, n_test, n_train)
            rows.append(
                {
                    "dropped": label,
                    "metric": metric,
                    "delta": d.mean(),
                    "lo": d.mean() - 1.96 * se,
                    "hi": d.mean() + 1.96 * se,
                }
            )
    return pd.DataFrame(rows)


def noise_importance(
    pipe: Pipeline,
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xte: pd.DataFrame,
    yte: np.ndarray,
    numeric,
    categorical,
    n_noise: int = 4,
    repeats: int = 30,
):
    """Permutation importance (drop in log loss) next to columns of pure noise added at fit time."""
    rng = np.random.default_rng(SEED)
    noise = [f"noise_{i}" for i in range(n_noise)]
    Xa, Xb = Xtr.copy(), Xte.copy()
    for c in noise:
        Xa[c] = rng.normal(size=len(Xa))
        Xb[c] = rng.normal(size=len(Xb))
    fitted = build_from(pipe, numeric + noise, categorical).fit(Xa, ytr)
    res = permutation_importance(
        fitted, Xb, yte, scoring="neg_log_loss", n_repeats=repeats, random_state=SEED, n_jobs=1
    )
    out = pd.DataFrame(
        {"feature": Xb.columns, "importance": res.importances_mean, "sd": res.importances_std}
    ).sort_values("importance", ascending=False)
    out["is_noise"] = out.feature.isin(noise)
    return out.reset_index(drop=True)


def build_from(pipe: Pipeline, numeric: list[str], categorical: list[str]) -> Pipeline:
    from .models import preprocessor

    return Pipeline([("pre", preprocessor(numeric, categorical)), ("model", clone(pipe.named_steps["model"]))])


def survival_check(
    train: pd.DataFrame, test: pd.DataFrame, numeric: list[str], categorical: list[str], tenure: list[str]
):
    """Cox and random survival forest on time-to-exit, scored by concordance and by AUC against the exit flag."""
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    from .data import survival_arrays
    from .models import preprocessor

    ev_tr, t_tr = survival_arrays(train)
    ev_te, t_te = survival_arrays(test)
    y_tr = Surv.from_arrays(ev_tr, t_tr)
    feats_all = [c for c in train.columns if c != TARGET]
    feats_clean = [c for c in feats_all if c not in tenure]
    rows = []
    risks = {}
    for set_name, feats in [("without tenure counters", feats_clean), ("with tenure counters", feats_all)]:
        num, cat = split_columns(feats, numeric, categorical)
        pre = preprocessor(num, cat)
        Ztr = pre.fit_transform(train[feats])
        Zte = pre.transform(test[feats])
        for mname, est in [
            ("cox", CoxPHSurvivalAnalysis(alpha=1.0)),
            (
                "survival forest",
                RandomSurvivalForest(n_estimators=200, min_samples_leaf=15, n_jobs=-1, random_state=SEED),
            ),
        ]:
            est.fit(Ztr, y_tr)
            r = est.predict(Zte)
            risks[(mname, set_name)] = r
            c = concordance_index_censored(ev_te, t_te, r)[0]
            rows.append(
                {
                    "model": mname,
                    "features": set_name,
                    "c_index": float(c),
                    "auc_vs_exit_flag": float(roc_auc_score(ev_te, r)),
                }
            )
    return pd.DataFrame(rows), risks


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)
