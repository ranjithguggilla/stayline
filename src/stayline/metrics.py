"""Scoring helpers: calibration error, bootstrap intervals, the corrected resampled standard error."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    total = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            total += len(idx) / len(y) * abs(y[idx].mean() - p[idx].mean())
    return float(total)


def score_all(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }


def bootstrap_ci(y, p, fn, n: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    while len(vals) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def corrected_se(fold_scores: np.ndarray, n_test: int, n_train: int) -> float:
    """Nadeau-Bengio standard error for repeated resampling, where folds overlap and plain SE is too small."""
    v = np.var(fold_scores, ddof=1)
    return float(np.sqrt(v * (1 / len(fold_scores) + n_test / n_train)))


def lift_at(y: np.ndarray, p: np.ndarray, frac: float) -> tuple[float, float, float]:
    """Return (precision, recall, lift) for the top `frac` of scores."""
    y = np.asarray(y)
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    hits = y[top].sum()
    precision = hits / k
    return float(precision), float(hits / y.sum()), float(precision / y.mean())
