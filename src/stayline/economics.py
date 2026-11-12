"""Retention-call economics: who to call when only a fraction of staff can be reached."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COSTS, Costs


def replacement_cost(monthly_income: np.ndarray, costs: Costs = COSTS) -> np.ndarray:
    return np.asarray(monthly_income, float) * 12 * costs.replacement_share_of_annual_pay


def net_value(
    leaves: np.ndarray, repl: np.ndarray, selected: np.ndarray, costs: Costs = COSTS, save_rate: float | None = None
) -> float:
    """Realised net value of calling the selected people: saved replacement cost minus conversation cost."""
    sr = costs.save_rate if save_rate is None else save_rate
    sel = np.asarray(selected, bool)
    return float((leaves[sel] * repl[sel]).sum() * sr - costs.cost_per_conversation * sel.sum())


def top_fraction(score: np.ndarray, frac: float) -> np.ndarray:
    k = int(round(frac * len(score)))
    mask = np.zeros(len(score), bool)
    if k > 0:
        mask[np.argsort(-score, kind="stable")[:k]] = True
    return mask


def strategies(p: np.ndarray, repl: np.ndarray, rule: np.ndarray, leaves: np.ndarray) -> dict[str, np.ndarray]:
    """Scores that each policy ranks by. 'oracle' ranks by realised value and is the ceiling."""
    return {
        "risk x replacement cost": p * repl,
        "risk only": p,
        "rule of thumb": rule + 1e-9 * p,
        "oracle (hindsight)": leaves * repl + 1e-9 * p,
    }


def policy_table(p, repl, rule, leaves, frac: float, costs: Costs = COSTS, n_boot: int = 1000, seed: int = 0):
    """Net value per 1,000 employees for each policy at a call budget, with bootstrap intervals over employees."""
    rng = np.random.default_rng(seed)
    scores = strategies(p, repl, rule, leaves)
    n = len(p)

    def per_1000(idx):
        out = {}
        for name, s in scores.items():
            sel = top_fraction(s[idx], frac)
            out[name] = net_value(leaves[idx], repl[idx], sel, costs) / len(idx) * 1000
        # random calls: expected value is the average over the group
        k = frac
        out["random calls"] = (
            k * (leaves[idx] * repl[idx]).mean() * costs.save_rate - k * costs.cost_per_conversation
        ) * 1000
        out["call everyone"] = net_value(leaves[idx], repl[idx], np.ones(len(idx), bool), costs) / len(idx) * 1000
        return out

    point = per_1000(np.arange(n))
    boots = [per_1000(rng.integers(0, n, n)) for _ in range(n_boot)]
    rows = []
    for name, v in point.items():
        b = np.array([x[name] for x in boots])
        rows.append({"policy": name, "net_per_1000": v, "lo": np.quantile(b, 0.025), "hi": np.quantile(b, 0.975)})
    return pd.DataFrame(rows).sort_values("net_per_1000", ascending=False).reset_index(drop=True)


def budget_curve(score, repl, leaves, fracs, costs: Costs = COSTS) -> pd.DataFrame:
    n = len(score)
    rows = []
    for f in fracs:
        sel = top_fraction(score, f)
        rows.append({"fraction": f, "net_per_1000": net_value(leaves, repl, sel, costs) / n * 1000})
    return pd.DataFrame(rows)


def break_even_save_rate(score, repl, leaves, frac, costs: Costs = COSTS) -> float:
    """Save rate at which the list at this budget stops paying for itself."""
    sel = top_fraction(score, frac)
    gross = (leaves[sel] * repl[sel]).sum()
    return float(costs.cost_per_conversation * sel.sum() / gross) if gross > 0 else float("inf")


def rule_of_thumb(df: pd.DataFrame, income_cut: float) -> np.ndarray:
    """The checklist a manager might use: overtime, single, low pay, no stock options, short tenure."""
    return (
        (
            (df["OverTime"] == "Yes").astype(int)
            + (df["MaritalStatus"] == "Single").astype(int)
            + (df["MonthlyIncome"] < income_cut).astype(int)
            + (df["StockOptionLevel"] == 0).astype(int)
            + (df["YearsAtCompany"] < 3).astype(int)
        )
        .to_numpy()
        .astype(float)
    )
